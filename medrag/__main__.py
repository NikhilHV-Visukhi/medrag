"""Command line: python -m medrag <command> [options]   (python -m medrag -h for help)"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime

from medrag.config import get_settings


def cmd_generate(a):
    from medrag.data import store
    from medrag.data.generate import generate

    t = generate(a.patients, a.mean_studies, a.dirty_rate, a.seed)
    store.write_all("raw", t)
    print("raw tables written:", {k: f"{len(v):,}" for k, v in t.items()})


def cmd_import_legacy(a):
    from medrag.data import store

    t = store.import_legacy_oads()
    store.write_all("raw", t)
    print("legacy oads tables imported into raw:", {k: f"{len(v):,}" for k, v in t.items()})


def cmd_check(a):
    from medrag.data import quality, store

    issues, md = quality.run(a.stage, store.read_all(a.stage), get_settings().reports_dir)
    print(md)
    print(f"\nSaved reports/quality_{a.stage}.md")
    if a.strict and any(i["severity"] == "error" for i in issues):
        sys.exit(1)


def cmd_clean(a):
    from medrag.data import quality, store
    from medrag.data.clean import clean

    s = get_settings()
    t, log = clean(store.read_all("raw"))
    store.write_all("clean", t)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    (s.reports_dir / "cleaning_log.md").write_text("# Cleaning log\n\n" + "\n".join(f"- {line}" for line in log), encoding="utf-8")
    print("\n".join(log))
    print("clean tables written:", {k: f"{len(v):,}" for k, v in t.items()})
    issues, _ = quality.run("clean", t, s.reports_dir)
    failing = [i for i in issues if i["severity"] == "error"]
    print(f"post-clean check: {len(failing)} error checks failing (reports/quality_clean.md)")


def cmd_documents(a):
    """Load outside documents -> check -> clean -> link to patients -> check again. Then run `index`."""
    from medrag.data import quality, store
    from medrag.documents import loaders, prepare

    s = get_settings()
    raw = loaders.load_all(s.data_dir / "documents")
    print("loaded:", raw["source"].value_counts().to_dict())
    store.write_table("raw", "documents", raw.assign(document_id=range(1, len(raw) + 1), patient_id=None, doc_date=None))
    before = prepare.check(raw)
    docs, log = prepare.clean(raw)
    docs, link_log = prepare.link_to_patients(docs, store.read_table("clean", "patients"))
    log += link_log
    store.write_table("clean", "documents", docs)
    after = prepare.check(docs)
    lines = ["# Document quality report", "", "| Check | Before | After |", "|---|---:|---:|"]
    lines += [f"| {b['check']} | {b['count']:,} | {c['count']:,} |" for b, c in zip(before, after)]
    lines += ["", "## Cleaning log", ""] + [f"- {x}" for x in log]
    lines += ["", "## Documents by source / specialty", "", docs.groupby("source")["specialty"].agg(lambda x: x.str.split(";").str[0].value_counts().head(12).to_dict()).to_string()]
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    (s.reports_dir / "quality_documents.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:len(before) + 4] + [""] + log))
    print(f"\n{len(docs):,} clean documents written. Next: python -m medrag index")


def cmd_index(a):
    from medrag.data import store
    from medrag.index.store import build
    from medrag.ingest.documents import build_chunks
    from medrag.providers.embeddings import get_embedder

    s = get_settings()
    t0 = time.time()
    chunks = build_chunks(store.read_all("clean"), a.max_words)
    print(f"{len(chunks):,} chunks ({(chunks['doc_type'] == 'study').sum():,} study, {(chunks['doc_type'] == 'patient').sum():,} patient) in {time.time() - t0:.0f}s")
    path = s.index_path(a.embed_provider, a.embed_model)
    manifest = build(chunks, get_embedder(a.embed_provider, a.embed_model), path)
    print(json.dumps(manifest, indent=2))


def cmd_ask(a):
    from medrag.providers.llms import get_llm
    from medrag.rag.answer import RAG
    from medrag.retrieval.retriever import Retriever

    rag = RAG(Retriever(embed_provider=a.embed_provider, embed_model=a.embed_model), get_llm(a.llm))
    questions = [a.question] if a.question else iter(lambda: input("\nQuestion (blank to quit): ").strip(), "")
    for q in questions:
        r, facts, stream = rag.stream(q, a.mode)
        print(f"\nfilters: {r.filters.describe()}")
        for i, row in enumerate(r.hits.itertuples(), 1):
            print(f"  [S{i}] {row.doc_id:<16} {row.via:<14} {row.patient_name} {row.text.splitlines()[1][:70] if row.doc_type == 'study' else 'profile'}")
        print()
        for piece in stream:
            print(piece, end="", flush=True)
        print()


def _raw_signature() -> str:
    """Changes on any insert, update or delete in the raw stage. Cheap: no table data is read."""
    from medrag.data import store

    s = get_settings()
    if s.data_backend == "parquet":
        sig = sorted((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in s.raw_dir.glob("*.parquet"))
    else:
        with store.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT relname, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables "
                        "WHERE schemaname = %s ORDER BY relname", (store.SCHEMAS["raw"],))
            sig = cur.fetchall()
    return hashlib.md5(json.dumps(sig, default=str).encode()).hexdigest()


def cmd_watch(a):
    """Keep the index in sync with the database: re-clean and re-index (incrementally) when raw data changes."""
    last = None
    while True:
        try:
            sig = _raw_signature()
            if sig != last:
                print(f"[{datetime.now():%H:%M:%S}] change detected -> clean + index")
                cmd_clean(a)
                cmd_index(a)
                last = sig
        except Exception as e:  # keep the watcher alive through transient DB errors
            print(f"[{datetime.now():%H:%M:%S}] sync failed: {e}")
        time.sleep(a.interval)


def cmd_eval(a):
    from medrag.eval import run, testset
    from medrag.index.store import Index
    from medrag.retrieval.retriever import Retriever

    s = get_settings()
    specs = [x.split(":", 1) for x in a.indexes.split(",")] if a.indexes else [(s.embed_provider, s.embed_model)]
    rerankers = a.rerankers.split(",") if a.rerankers else [s.rerank_model]
    indexes = {f"{p}:{m}": Index(s.index_path(p, m)) for p, m in specs}
    main_name, main_index = next(iter(indexes.items()))

    configs = []
    for name, idx in indexes.items():  # every index with the first reranker
        configs.append(run.Config(f"{name} | hybrid+rerank ({rerankers[0].split('/')[-1]})", Retriever(idx, rerank_model=rerankers[0])))
    for rr in rerankers[1:]:  # every other reranker on the main index
        configs.append(run.Config(f"{main_name} | hybrid+rerank ({rr.split('/')[-1]})", Retriever(main_index, rerank_model=rr)))
    if a.blends:  # also each reranker with its order replacing (not fused with) the hybrid order
        for rr in rerankers:
            configs.append(run.Config(f"{main_name} | hybrid+rerank ({rr.split('/')[-1]}, replace)",
                                      Retriever(main_index, rerank_model=rr, rerank_blend="replace")))
    main = configs[0].retriever
    if a.baselines:  # ablations on the main configuration
        configs += [run.Config(f"{main_name} | {m}", main, mode=m) for m in ("hybrid", "dense", "bm25")]
        configs += [run.Config(f"{main_name} | hybrid+rerank, no query expansion", main, expand=False),
                    run.Config(f"{main_name} | hybrid+rerank, no filters", main, use_filters=False)]

    out = s.reports_dir / f"eval_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    items = testset.build(main.meta, a.per_type)
    if a.difficulty != "all":
        items = [i for i in items if i["difficulty"] == a.difficulty]
    testset.save(items, out / "testset.json")
    testset.review_sheet(items, main.meta, out / "golden_review.csv")
    print(f"{len(items)} test questions -> {out / 'testset.json'} (hand-check sample: golden_review.csv)")

    ret_df = run.eval_retrieval(items, configs, a.k)
    gen_df = None
    if not a.skip_generation:
        llms = a.llms.split(",") if a.llms else [f"{s.llm_provider}:{s.llm_model}"]
        by_type: dict[str, list] = {}
        for it in items:
            by_type.setdefault(it["type"], []).append(it)
        per = max(1, a.max_answers // len(by_type))  # stratified: the same number of answers from every question type
        gen_items = [it for group in by_type.values() for it in group[:per]]
        gen_df = run.eval_generation(gen_items, main, llms, a.judge)
    print(run.report(ret_df, gen_df, out, a.k))
    print(f"\nSaved {out}")


def cmd_pipeline(a):
    cmd_generate(a)
    a.stage, a.strict = "raw", False
    cmd_check(a)
    cmd_clean(a)
    cmd_index(a)


def main():
    ap = argparse.ArgumentParser(prog="medrag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def gen_args(p):
        p.add_argument("--patients", type=int, default=20_000)
        p.add_argument("--mean-studies", type=float, default=5.0)
        p.add_argument("--dirty-rate", type=float, default=0.03, help="share of rows given realistic data errors")
        p.add_argument("--seed", type=int, default=7)

    def index_args(p):
        p.add_argument("--embed-provider")
        p.add_argument("--embed-model")
        p.add_argument("--max-words", type=int, default=220)

    p = sub.add_parser("generate", help="generate a large synthetic dataset into the raw stage")
    gen_args(p)
    p.set_defaults(fn=cmd_generate)
    sub.add_parser("import-legacy", help="import the original oads.* tables into the raw stage").set_defaults(fn=cmd_import_legacy)
    p = sub.add_parser("check", help="data quality report")
    p.add_argument("--stage", choices=["raw", "clean"], default="raw")
    p.add_argument("--strict", action="store_true", help="exit 1 if any error-level check fails")
    p.set_defaults(fn=cmd_check)
    sub.add_parser("clean", help="clean raw -> clean stage").set_defaults(fn=cmd_clean)
    sub.add_parser("documents", help="ingest outside documents (Open-i, MTSamples): check, clean, link to patients").set_defaults(fn=cmd_documents)
    p = sub.add_parser("index", help="chunk + embed + build hybrid index from clean data")
    index_args(p)
    p.set_defaults(fn=cmd_index)
    p = sub.add_parser("ask", help="ask a question (interactive if no question given)")
    p.add_argument("question", nargs="?")
    p.add_argument("--llm", help="provider:model, e.g. anthropic:claude-opus-5-5")
    p.add_argument("--mode", default="hybrid+rerank")
    p.add_argument("--embed-provider")
    p.add_argument("--embed-model")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("watch", help="poll the database and keep the index in sync")
    p.add_argument("--interval", type=int, default=30)
    index_args(p)
    p.set_defaults(fn=cmd_watch)
    p = sub.add_parser("eval", help="evaluate retrieval and answers, compare models")
    p.add_argument("--indexes", help="comma list of provider:model indexes to compare")
    p.add_argument("--llms", help="comma list of provider:model LLMs to compare")
    p.add_argument("--per-type", type=int, default=20)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max-answers", type=int, default=40, help="questions used for the (slower) answer evaluation")
    p.add_argument("--rerankers", help="comma list of cross-encoder models to compare (first one is the main)")
    p.add_argument("--baselines", action="store_true", help="also run ablations: hybrid, dense, bm25, no expansion, no filters")
    p.add_argument("--blends", action="store_true", help="also run every reranker in 'replace' mode (default is 'fuse')")
    p.add_argument("--difficulty", choices=["all", "easy", "hard", "documents"], default="all")
    p.add_argument("--judge", action="store_true", help="also score answers with the judge LLM from .env")
    p.add_argument("--skip-generation", action="store_true")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("pipeline", help="generate -> check -> clean -> index in one go")
    gen_args(p)
    index_args(p)
    p.set_defaults(fn=cmd_pipeline)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
