"""Chat LLM providers behind one interface: stream(system, user) yields text; complete() returns it all."""
import json
from collections.abc import Iterator

from medrag.config import get_settings

# Claude models that accept the server-side refusal fallback ("default" routing).
CLAUDE_FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")


class LLM:
    provider: str

    def __init__(self, model: str):
        self.model = model

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def stream(self, system: str, user: str, max_tokens: int = 1500) -> Iterator[str]:
        raise NotImplementedError

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        return "".join(self.stream(system, user, max_tokens))


class OllamaLLM(LLM):
    provider = "ollama"

    def stream(self, system, user, max_tokens=1500):
        import httpx

        body = {
            "model": self.model, "stream": True, "think": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": max_tokens},
        }
        in_think = False
        with httpx.stream("POST", f"{get_settings().ollama_url}/api/chat", json=body, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                piece = json.loads(line).get("message", {}).get("content", "")
                # Drop any <think>...</think> reasoning a model still emits into the answer.
                if "<think>" in piece:
                    in_think, piece = True, piece.split("<think>")[0]
                if in_think and "</think>" in piece:
                    in_think, piece = False, piece.split("</think>", 1)[1]
                if piece and not in_think:
                    yield piece


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, model):
        super().__init__(model)
        import anthropic

        self.client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key or None)

    def stream(self, system, user, max_tokens=1500):
        kwargs = dict(
            model=self.model, max_tokens=max(max_tokens, 4000), system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "medium"},
        )
        if self.model in CLAUDE_FALLBACK_MODELS:
            # On a safety decline the API re-runs the request on a fallback model in the same call.
            kwargs |= dict(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        with self.client.beta.messages.stream(**kwargs) as s:
            yield from s.text_stream
            final = s.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n[The model declined to answer this request.]"


class OpenAILLM(LLM):
    provider = "openai"

    def __init__(self, model):
        super().__init__(model)
        from openai import OpenAI

        self.client = OpenAI(api_key=get_settings().openai_api_key or None)

    def stream(self, system, user, max_tokens=1500):
        resp = self.client.chat.completions.create(
            model=self.model, stream=True, max_completion_tokens=max(max_tokens, 4000),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        for ev in resp:
            if ev.choices and ev.choices[0].delta.content:
                yield ev.choices[0].delta.content


class GroqLLM(OpenAILLM):
    """Groq: free tier, OpenAI-compatible API (e.g. llama-3.3-70b-versatile). Key: GROQ_API_KEY."""
    provider = "groq"
    base_url = "https://api.groq.com/openai/v1"
    key_field = "groq_api_key"

    def __init__(self, model):
        LLM.__init__(self, model)
        from openai import OpenAI

        self.client = OpenAI(api_key=getattr(get_settings(), self.key_field) or "missing", base_url=self.base_url)

    def stream(self, system, user, max_tokens=1500):
        resp = self.client.chat.completions.create(
            model=self.model, stream=True, max_tokens=max(max_tokens, 2000), temperature=0.1,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        for ev in resp:
            if ev.choices and ev.choices[0].delta.content:
                yield ev.choices[0].delta.content


class OpenRouterLLM(GroqLLM):
    """OpenRouter: many models, some free (model ids ending in ':free'). Key: OPENROUTER_API_KEY."""
    provider = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    key_field = "openrouter_api_key"


class GeminiLLM(LLM):
    provider = "gemini"

    def __init__(self, model):
        super().__init__(model)
        from google import genai

        self.client = genai.Client(api_key=get_settings().gemini_api_key or None)

    def stream(self, system, user, max_tokens=1500):
        from google.genai import types

        cfg = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max(max_tokens, 4000), temperature=0.1)
        for ev in self.client.models.generate_content_stream(model=self.model, contents=user, config=cfg):
            if ev.text:
                yield ev.text


PROVIDERS = {c.provider: c for c in (OllamaLLM, AnthropicLLM, OpenAILLM, GroqLLM, OpenRouterLLM, GeminiLLM)}


def get_llm(spec: str | None = None) -> LLM:
    """spec is 'provider:model' (e.g. 'ollama:qwen2.5:3b', 'anthropic:claude-opus-5-5'); default from .env."""
    s = get_settings()
    if spec:
        provider, model = spec.split(":", 1)
    else:
        provider, model = s.llm_provider, s.llm_model
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider {provider!r}; choose from {sorted(PROVIDERS)}")
    return PROVIDERS[provider](model)


def get_judge() -> LLM:
    s = get_settings()
    return get_llm(f"{s.judge_provider}:{s.judge_model}")
