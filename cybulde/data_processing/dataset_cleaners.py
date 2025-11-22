from __future__ import annotations

import re
import string

from abc import ABC, abstractmethod
from functools import lru_cache

import nltk

from nltk.tokenize import word_tokenize

from cybulde.utils.utils import SpellCorrectionModel


@lru_cache(maxsize=1)
def _ensure_nltk_and_get_stopwords() -> set[str]:
    """NLTK kaynaklarını indir ve İngilizce stopword set'ini döndür.

    Dask ile uyumlu olması için:
    - Sadece düz `set[str]` döndürürüz (LazyCorpusLoader vs yok).
    - NLTK resource kontrollerini `nltk.data.find` ile yaparız.
    """
    # Gerekli paketler kurulu mu diye bak; yoksa indir
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Yeni NLTK sürümlerinde çıkan `punkt_tab` hatası için
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            # Eski sürümlerde bu paket yok, sessizce geç
            pass

    # Burada import etmek güvenli, sadece bu fonksiyon içinde kalacak
    from nltk.corpus import stopwords as nltk_stopwords

    return set(nltk_stopwords.words("english"))


class DatasetCleaner(ABC):
    def __call__(self, text: str | list[str]) -> str | list[str]:
        if isinstance(text, str):
            return self.clean_text(text)
        return self.clean_words(text)

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """
        Cleans the given string
        """

    @abstractmethod
    def clean_words(self, words: list[str]) -> list[str]:
        """
        Cleans each word in a list of word
        """


class StopWordsDatasetCleaner(DatasetCleaner):
    def __init__(self) -> None:
        super().__init__()
        # Burada NLTK'yi hazırlarız ve düz Python set'i alırız
        self.stopwords = _ensure_nltk_and_get_stopwords()

    def clean_text(self, text: str) -> str:
        # Tokenization için NLTK'nin word_tokenize'ını kullanıyoruz
        tokens = word_tokenize(text)
        cleaned_tokens = [t for t in tokens if t.lower() not in self.stopwords]
        return " ".join(cleaned_tokens)

    def clean_words(self, words: list[str]) -> list[str]:
        # words zaten token listesi ise, tekrar tokenize etmeye gerek yok
        return [w for w in words if w.lower() not in self.stopwords]


class ToLowerCaseDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return text.lower()

    def clean_words(self, words: list[str]) -> list[str]:
        return [word.lower() for word in words]


class URLDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return re.sub(r"http\S+", "", text, flags=re.MULTILINE)

    def clean_words(self, words: list[str]) -> list[str]:
        return [self.clean_text(word) for word in words]


class PunctuationDatasetCleaner(DatasetCleaner):
    def __init__(self, punctuation: str = string.punctuation) -> None:
        super().__init__()
        self.table = str.maketrans("", "", punctuation)

    def clean_text(self, text: str) -> str:
        return " ".join(self.clean_words(text.split()))

    def clean_words(self, words: list[str]) -> list[str]:
        return [word.translate(self.table) for word in words if word.translate(self.table)]


class NonLettersDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return " ".join(self.clean_words(text.split()))

    def clean_words(self, words: list[str]) -> list[str]:
        return [word for word in words if word.isalpha()]


class NewLineCharacterDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return text.replace("\n", "")

    def clean_words(self, words: list[str]) -> list[str]:
        return [self.clean_text(word) for word in words]


class NonASCIIDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return " ".join(self.clean_words(text.split()))

    def clean_words(self, words: list[str]) -> list[str]:
        return [word for word in words if word.isascii()]


class ReferenceToAccountDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return re.sub(r"@\w", "", text)

    def clean_words(self, words: list[str]) -> list[str]:
        text = " ".join(words)
        return self.clean_text(text).split()


class ReTweetDatasetCleaner(DatasetCleaner):
    def clean_text(self, text: str) -> str:
        return re.sub(r"\bRT\b", "", text, flags=re.IGNORECASE)

    def clean_words(self, words: list[str]) -> list[str]:
        text = " ".join(words)
        return self.clean_text(text).split()


class SpellCorrectionDatasetCleaner(DatasetCleaner):
    def __init__(self, spell_correction_model: SpellCorrectionModel) -> None:
        super().__init__()
        self.spell_correction_model = spell_correction_model

    def clean_text(self, text: str) -> str:
        return self.spell_correction_model(text)

    def clean_words(self, words: list[str]) -> list[str]:
        text = " ".join(words)
        return self.clean_text(text).split()


class CharacterLimiterDatasetCleaner(DatasetCleaner):
    def __init__(self, character_limit: int = 300) -> None:
        super().__init__()
        self.character_limit = character_limit

    def clean_text(self, text: str) -> str:
        return text[: self.character_limit]

    def clean_words(self, words: list[str]) -> list[str]:
        text = " ".join(words)
        return self.clean_text(text).split()


class DatasetCleanerManager:
    def __init__(self, dataset_cleaners: dict[str, DatasetCleaner]) -> None:
        self.dataset_cleaners = dataset_cleaners

    def __call__(self, text: str | list[str]) -> str | list[str]:
        for dataset_cleaner in self.dataset_cleaners.values():
            text = dataset_cleaner(text)
        return text
