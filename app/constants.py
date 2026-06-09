# Roles are free-form strings. ALLOWED_ROLES is a UI / ASR-hotword hint only and is
# NOT enforced for validation anywhere in the backend. Any non-blank role string is valid.
ALLOWED_ROLES = [
    "AI Engineer",
    "Generative AI Engineer",
    "GenAI Engineer",
    "LLM Engineer",
    "RAG Engineer",
    "AI Systems Engineer",
    "ML Engineer",
    "Computer Vision Engineer",
    "Agentic AI Engineer",
    "Data Science",
    "Prompt Engineer",
    "Platform Engineer",
    "GET",
    "AI Product Engineer",
]

ALLOWED_EMPLOYMENT_TYPES = [
    "Internship",
    "Full Time",
    "Part Time",
]

ALLOWED_LOCATIONS = [
    "remote",
    "hybrid",
    "onsite",
]

ALLOWED_CURRENT_STAGES = [
    "Tailored",
    "Applied",
    "Networked",
    "Engaged",
    "COLD_MAIL",
    "Followed up",
]

ALLOWED_PRIORITIES = [
    "LOW",
    "MEDIUM",
    "HIGH",
]

STATUS_OPTIONS = [
    "Interested",
    "Applied",
    "Rejected",
    "Interview",
    "Offer",
    "Archived",
]

EVENT_TYPES = {
    "application_saved",
    "field_changed",
    "note_added",
    "status_changed",
    "application_archived",
    "application_restored",
}
