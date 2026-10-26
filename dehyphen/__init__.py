from importlib.metadata import version

from .format import (
    format_to_paragraph,
    format_to_text,
    paragraph_to_format,
    text_to_format,
)
from .scorer import FlairScorer

__version__ = version("dehyphen")

__all__ = [
    "FlairScorer",
    "format_to_paragraph",
    "format_to_text",
    "paragraph_to_format",
    "text_to_format",
]
