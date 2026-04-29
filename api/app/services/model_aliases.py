DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_MODEL_ALIASES = {
    "gemini-1.5-flash": DEFAULT_GEMINI_MODEL,
    "gemini-1.5-flash-latest": DEFAULT_GEMINI_MODEL,
}


def normalize_model_name(model_name: str) -> str:
    normalized = (model_name or "").strip()
    if not normalized:
        return normalized
    return _MODEL_ALIASES.get(normalized, normalized)
