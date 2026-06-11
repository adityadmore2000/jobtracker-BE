import re
import string


WHITESPACE_RE = re.compile(r"\s+")
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_company_name(value: str) -> str:
    trimmed = value.strip().lower()
    collapsed = WHITESPACE_RE.sub(" ", trimmed)
    separated_punctuation = collapsed.translate(str.maketrans({char: " " for char in string.punctuation}))
    without_punctuation = separated_punctuation.translate(PUNCT_TRANSLATION)
    return WHITESPACE_RE.sub(" ", without_punctuation).strip()


def has_meaningful_company_difference(left: str, right: str) -> bool:
    return normalize_company_name(left) != normalize_company_name(right)
