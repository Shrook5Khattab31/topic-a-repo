"""
Text normalization levers for Ticket 2.

Each function is an independent, togglable transformation so experiments can
isolate the effect of a single decision (URL handling, mentions, hashtags,
punctuation, casing, emoji) rather than changing several things at once.
"""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
PUNCT_RE = re.compile(r"[^\w\s]")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def normalize(
    text: str,
    lowercase: bool = True,
    strip_urls: bool = True,
    strip_mentions: bool = True,
    unwrap_hashtags: bool = True,
    strip_punct: bool = False,
    strip_emoji: bool = False,
) -> str:
    """Apply a configurable subset of normalization steps.
    Defaults mirror a common/reasonable baseline preprocessing choice."""
    t = text

    if strip_urls:
        t = URL_RE.sub(" ", t)
    if strip_mentions:
        t = MENTION_RE.sub(" ", t)
    if unwrap_hashtags:
        t = HASHTAG_RE.sub(r"\1", t)  # keep the word, drop the '#'
    if strip_emoji:
        t = EMOJI_RE.sub(" ", t)
    if strip_punct:
        t = PUNCT_RE.sub(" ", t)
    if lowercase:
        t = t.lower()

    t = re.sub(r"\s+", " ", t).strip()
    return t


# Named configs used across tickets 1 and 2 so results are reproducible and
# comparable. Add/rename configs here rather than passing ad-hoc kwargs
# around scripts.
CONFIGS = {
    "raw": dict(lowercase=False, strip_urls=False, strip_mentions=False,
                unwrap_hashtags=False, strip_punct=False, strip_emoji=False),
    "baseline": dict(lowercase=True, strip_urls=True, strip_mentions=True,
                      unwrap_hashtags=True, strip_punct=False, strip_emoji=False),
    "aggressive": dict(lowercase=True, strip_urls=True, strip_mentions=True,
                        unwrap_hashtags=True, strip_punct=True, strip_emoji=True),
    "keep_hashtags_raw": dict(lowercase=True, strip_urls=True, strip_mentions=True,
                               unwrap_hashtags=False, strip_punct=False, strip_emoji=False),
    "keep_case": dict(lowercase=False, strip_urls=True, strip_mentions=True,
                       unwrap_hashtags=True, strip_punct=False, strip_emoji=False),
}


def apply_config(text: str, config_name: str) -> str:
    return normalize(text, **CONFIGS[config_name])
