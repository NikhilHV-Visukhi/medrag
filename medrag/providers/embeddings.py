"""Embedding providers behind one interface: embed_documents / embed_query -> L2-normalised float32."""
import hashlib
import re
from collections import OrderedDict

import numpy as np

from medrag.config import get_settings

# Models trained with asymmetric query/document instructions.
PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
    "snowflake-arctic-embed": ("Represent this sentence for searching relevant passages: ", ""),
}


def _prefixes(model: str) -> tuple[str, str]:
    m = model.lower()
    for key, val in PREFIXES.items():
        if key in m:
            return val
    return "", ""


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


class Embedder:
    name: str

    def __init__(self, model: str):
        self.model = model
        self.q_prefix, self.d_prefix = _prefixes(model)

    def _embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._embed([self.d_prefix + t for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._embed([self.q_prefix + text]))[0]


def _cached_query(embed_query):
    """Wrap an embed_query method with a small LRU cache: repeated questions skip the model."""
    def wrapper(self, text):
        cache = self.__dict__.setdefault("_qcache", OrderedDict())
        if text in cache:
            cache.move_to_end(text)
            return cache[text]
        v = cache[text] = embed_query(self, text)
        if len(cache) > 1024:
            cache.popitem(last=False)
        return v
    return wrapper


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str):
        super().__init__(model)
        from sentence_transformers import SentenceTransformer

        self.st = SentenceTransformer(model)

    def _embed(self, texts):
        return self.st.encode(texts, batch_size=get_settings().embed_batch_size, show_progress_bar=False)


class MedCPTEmbedder(Embedder):
    """NCBI MedCPT: two encoders trained on PubMed search logs, one for queries and one for documents
    (the same family as the MedCPT reranker). Model name is ignored; the [CLS] vector is the embedding."""
    name = "medcpt"

    def __init__(self, model: str = "ncbi/MedCPT"):
        super().__init__(model)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enc = {kind: (AutoTokenizer.from_pretrained(f"ncbi/MedCPT-{kind}-Encoder"),
                           AutoModel.from_pretrained(f"ncbi/MedCPT-{kind}-Encoder").to(self.device).eval())
                    for kind in ("Query", "Article")}

    def _encode(self, kind: str, texts: list[str], max_len: int) -> np.ndarray:
        tok, model = self.enc[kind]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), get_settings().embed_batch_size):
                x = tok(texts[i:i + get_settings().embed_batch_size], truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(self.device)
                out.append(model(**x).last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(out)

    def embed_documents(self, texts):
        return _normalise(self._encode("Article", texts, 512))

    def embed_query(self, text):
        return _normalise(self._encode("Query", [text], 64))[0]


class OllamaEmbedder(Embedder):
    name = "ollama"

    def _embed(self, texts):
        import httpx

        r = httpx.post(f"{get_settings().ollama_url}/api/embed", json={"model": self.model, "input": texts}, timeout=600)
        r.raise_for_status()
        return np.array(r.json()["embeddings"])


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, model):
        super().__init__(model)
        from openai import OpenAI

        self.client = OpenAI(api_key=get_settings().openai_api_key or None)

    def _embed(self, texts):
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data])


class GeminiEmbedder(Embedder):
    name = "gemini"

    def __init__(self, model):
        super().__init__(model)
        from google import genai

        self.client = genai.Client(api_key=get_settings().gemini_api_key or None)
        self._task = "RETRIEVAL_DOCUMENT"

    def embed_query(self, text):
        self._task = "RETRIEVAL_QUERY"
        try:
            return super().embed_query(text)
        finally:
            self._task = "RETRIEVAL_DOCUMENT"

    def _embed(self, texts):
        from google.genai import types

        resp = self.client.models.embed_content(model=self.model, contents=texts, config=types.EmbedContentConfig(task_type=self._task))
        return np.array([e.values for e in resp.embeddings])


class HashEmbedder(Embedder):
    """Deterministic bag-of-words hashing. No download, no quality: for tests and smoke runs only."""
    name = "hash"
    dim = 512

    def _embed(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        return out


Embedder.embed_query = _cached_query(Embedder.embed_query)
MedCPTEmbedder.embed_query = _cached_query(MedCPTEmbedder.embed_query)
PROVIDERS = {c.name: c for c in (SentenceTransformerEmbedder, MedCPTEmbedder, OllamaEmbedder, OpenAIEmbedder, GeminiEmbedder, HashEmbedder)}
_cache: dict[tuple, Embedder] = {}


def get_embedder(provider: str | None = None, model: str | None = None) -> Embedder:
    s = get_settings()
    key = (provider or s.embed_provider, model or s.embed_model)
    if key not in _cache:
        if key[0] not in PROVIDERS:
            raise ValueError(f"Unknown embedding provider {key[0]!r}; choose from {sorted(PROVIDERS)}")
        _cache[key] = PROVIDERS[key[0]](key[1])
    return _cache[key]
