from .constants import ALLOWED_ROLES


def normalize_lookup_text(value: str) -> str:
    return " ".join(value.replace("-", " ").replace("_", " ").strip().casefold().split())


ROLE_ALIASES = {
    "ai engineer": "AI Engineer",
    "ai eng": "AI Engineer",
    "generative ai engineer": "Generative AI Engineer",
    "genai": "Generative AI Engineer",
    "gen ai": "Generative AI Engineer",
    "genai engineer": "Generative AI Engineer",
    "llm engineer": "LLM Engineer",
    "rag": "RAG Engineer",
    "rag engineer": "RAG Engineer",
    "ai systems engineer": "AI Systems Engineer",
    "ml engineer": "ML Engineer",
    "machine learning engineer": "ML Engineer",
    "cv engineer": "Computer Vision Engineer",
    "computer vision engineer": "Computer Vision Engineer",
    "agentic ai engineer": "Agentic AI Engineer",
    "data science": "Data Science",
    "prompt engineer": "Prompt Engineer",
    "platform engineer": "Platform Engineer",
    "get": "GET",
    "graduate engineer trainee": "GET",
    "ai product engineer": "AI Product Engineer",
}

ROLE_LOOKUP = {normalize_lookup_text(role): role for role in ALLOWED_ROLES}
ROLE_LOOKUP.update({normalize_lookup_text(alias): role for alias, role in ROLE_ALIASES.items()})


def canonicalize_role(value: str | None) -> str | None:
    if value is None:
        return None
    return ROLE_LOOKUP.get(normalize_lookup_text(value))
