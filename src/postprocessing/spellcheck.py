"""Spell checking utilities.

This module uses `pyspellchecker` (module name `spellchecker`).
If the package is not installed in the runtime environment, a small noop
fallback is provided so that the server remains functional and Pylance
won't crash while providing a clear warning.
"""

try:
    from spellchecker import SpellChecker  # pyspellchecker package
    # Silence Pylance missing-import lint if package is not present in environment
    # (useful for dev environments where user hasn't installed dependencies yet)
    # type: ignore[reportMissingImports]
    _SPELLCHECKER_AVAILABLE = True
except Exception:
    SpellChecker = None  # type: ignore
    _SPELLCHECKER_AVAILABLE = False

import re
import warnings

class SpellCorrector:
    def __init__(self, language='en', custom_word_list=None, case_sensitive=False):
        """Initialize the SpellCorrector.

        Args:
            language: Language to load in SpellChecker (default 'en').
            custom_word_list: Optional iterable of words to extend the dictionary.
            case_sensitive: Leave as False for speed (lowercase everything).
        """
        self.case_sensitive = case_sensitive
        if not _SPELLCHECKER_AVAILABLE:
            # Create a very small fallback object to keep the API stable
            warnings.warn("pyspellchecker not installed, SpellCorrector will be a no-op (no corrections).", ImportWarning)
            self.spell = None
        else:
            self.spell = SpellChecker(language=language)
        if custom_word_list is not None:
            try:
                self.spell.word_frequency.load_words(custom_word_list)
            except Exception:
                # The library expects a list of str. If single str passed, ignore
                pass

    def correct_word(self, word: str) -> str:
        if not word:
            return word
        if not self.case_sensitive:
            w = word.lower()
        else:
            w = word
        # If no spellchecker backend available, just return the original
        if not _SPELLCHECKER_AVAILABLE or self.spell is None:
            return word

        # If word in dictionary, no change
        if w in self.spell:
            return word

        corrected = self.spell.correction(w)
        return corrected if corrected else word

    def correct_text(self, text: str) -> str:
        """Correct the input text word-by-word using pyspellchecker.

        Behavior:
            - Splits text by whitespace, attempts to correct each token.
            - If the model output has no spaces, attempts to correct the whole string as one token.
            - Non-alphabetic tokens (numbers) are left unchanged except as part of words.
        """
        text = text.strip()
        if not text:
            return text

        # If there are white spaces, treat as multiple words
        if re.search(r"\s", text):
            tokens = text.split()
            corrected_tokens = [self.correct_word(tok) for tok in tokens]
            return ' '.join(corrected_tokens)
        else:
            # Single token - try to correct as a whole
            corrected = self.correct_word(text)
            return corrected if corrected else text


# Utility function for quick usage
_default_corrector = None

def correct_text(text: str, language='en', custom_word_list=None):
    global _default_corrector
    if _default_corrector is None:
        _default_corrector = SpellCorrector(language=language, custom_word_list=custom_word_list)
    return _default_corrector.correct_text(text)


def is_spellchecker_available() -> bool:
    """Return whether the pyspellchecker backend is available in the runtime.

    Useful in diagnostics and for fallback behavior in unit tests.
    """
    return _SPELLCHECKER_AVAILABLE
