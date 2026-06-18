"""VoiceTut-TTS — Egyptian-Arabic & code-switching text-to-speech.

Built on a fine-tuned OmniVoice checkpoint. Companion to QwenCleo-ASR.
"""

from .engine import (
    VoiceTutTTS,
    GenerationParams,
    split_sentences,
    resolve_language,
    DEFAULT_REPO,
)
from .normalization import ArabicNormalizer, NormalizerConfig, number_to_arabic_words
from .speakers import Speaker, SpeakerRegistry

__version__ = "0.1.0"

__all__ = [
    "VoiceTutTTS",
    "GenerationParams",
    "ArabicNormalizer",
    "NormalizerConfig",
    "number_to_arabic_words",
    "Speaker",
    "SpeakerRegistry",
    "split_sentences",
    "resolve_language",
    "DEFAULT_REPO",
    "__version__",
]
