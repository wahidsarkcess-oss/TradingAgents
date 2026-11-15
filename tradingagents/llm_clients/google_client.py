import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient, normalize_content
from .retrying import call_with_budget_retries
from .validators import validate_model

_GEMINI_VERSION = re.compile(r"^gemini-(\d+)\.(\d+)")


def _accepts_minimal_thinking(model: str) -> bool:
    """Whether ``thinking_level="minimal"`` is accepted: numbered Flash models
    before 3.8. Pro, 3.8+ and version-less aliases (which move between
    generations) are treated as rejecting it."""
    model_lc = model.lower()
    match = _GEMINI_VERSION.match(model_lc)
    return bool(match) and "pro" not in model_lc and (
        (int(match.group(1)), int(match.group(2))) < (3, 8)
    )


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output.

    Gemini 3 models return content as list of typed blocks.
    This normalizes to string for consistent downstream handling.
    """

    # Set under budget limits: the SDK's own retries are off and this many are
    # made here, each approved by the budget first. None: the SDK retries as before.
    budget_retries: int | None = None

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        parent = super()._generate
        if self.budget_retries is None:
            return parent(messages, stop, run_manager, **kwargs)
        return call_with_budget_retries(lambda: parent(messages, stop, run_manager, **kwargs),
                                        self.budget_retries)


# The read timeout the OpenAI and Anthropic SDKs use by default.
REQUEST_TIMEOUT_SECONDS = 600


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in ("timeout", "max_retries", "temperature", "max_output_tokens",
                    "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        # Without a timeout a stalled Gemini call waits forever; the OpenAI and
        # Anthropic SDKs give up after the same 600 seconds (#1417).
        llm_kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)

        # Unified api_key maps to provider-specific google_api_key
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        # Gemini 3.x takes the string ``thinking_level`` (the integer
        # ``thinking_budget`` was for the now-retired 2.5 line). Pro, Gemini
        # 3.8+ and the -latest aliases reject "minimal" with a 400; "low" is
        # accepted everywhere, so it is the fallback.
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            if thinking_level == "minimal" and not _accepts_minimal_thinking(self.model):
                thinking_level = "low"
            llm_kwargs["thinking_level"] = thinking_level

        # Under budget limits the google-genai SDK makes exactly one attempt
        # (attempts=1) and the retries happen in _generate, through the budget.
        budget_retries = self.kwargs.get("budget_controlled_retries")
        if budget_retries is not None:
            llm_kwargs["max_retries"] = 1
            llm_kwargs["budget_retries"] = int(budget_retries)

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
