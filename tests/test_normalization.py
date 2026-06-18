"""Unit tests for text normalization and engine text-logic (no model/GPU needed)."""

import pytest

from voicetut_tts.normalization import (
    ArabicNormalizer, number_to_arabic_words, _say_time, _say_phone_number,
)
from voicetut_tts.engine import split_sentences, resolve_language


# --------------------------------------------------------------- numbers
@pytest.mark.parametrize("n,expected", [
    (0, "صفر"), (7, "سبعة"), (15, "خمستاشر"), (21, "واحد وعشرين"),
    (100, "مية"), (250, "ميتين وخمسين"), (1000, "ألف"),
])
def test_number_words(n, expected):
    assert number_to_arabic_words(n) == expected


# --------------------------------------------------------------- normalizer
@pytest.fixture(scope="module")
def norm():
    return ArabicNormalizer()


def test_abbrev_not_inside_word(norm):
    # "م" must NOT expand inside "محمد"
    assert "محمد" in norm("محمد راجل كويس")


def test_currency(norm):
    assert "جنيه" in norm("الكتاب بـ 250 جنيه")
    assert "دولار" in norm("سعره 75$")


def test_percent(norm):
    assert "في المية" in norm("خصم 25%")


@pytest.mark.parametrize("h,m,expected", [
    (3, 30, "تلاتة و نص"),
    (4, 15, "أربعة و ربع"),
    (5, 20, "خمسة و تلت"),
    (5, 25, "خمسة و نص الا خمسة"),
    (6, 10, "ستة و عشرة"),
    (7, 45, "تمانية الا ربع"),
    (9, 50, "عشرة الا عشرة"),
    (11, 55, "اتناشر الا خمسة"),
    (1, 5, "واحدة و خمسة"),
    (12, 0, "اتناشر"),
])
def test_time_colloquial(h, m, expected):
    assert _say_time(h, m) == expected


@pytest.mark.parametrize("prefix", ["زيرو عشرة", "زيرو حداشر", "زيرو اتناشر", "زيرو خمستاشر"])
def test_phone_prefixes(prefix):
    pre_digit = {"عشرة": "010", "حداشر": "011", "اتناشر": "012", "خمستاشر": "015"}
    code = pre_digit[prefix.split()[-1]]
    out = _say_phone_number(code + "47450629")
    assert out.strip().startswith(prefix)


def test_phone_pairs_as_tens():
    # 011 | 47 | 45 | 06 | 29
    out = _say_phone_number("01147450629")
    assert "حداشر" in out and "سبعة وأربعين" in out and "تسعة وعشرين" in out


def test_name_transliteration():
    n = ArabicNormalizer()
    assert "أحمد" in n("Ahmed")
    assert "محمد" in n("Mohamed")
    assert "منى" in n("Mona")


def test_custom_names():
    n = ArabicNormalizer()
    n.add_names({"Ziad": "زياد"})
    assert "زياد" in n("Ziad")


def test_email_url(norm):
    out = norm("ابعت على a.b@gmail.com")
    # email is spelled out: the "@" and "." are read aloud and the raw token is gone
    assert "@" not in out and "a.b@gmail.com" not in out
    assert ("at" in out or "آت" in out) and ("dot" in out or "نقطة" in out)


def test_keeps_english(norm):
    assert "meeting" in norm("عندي meeting بكرة")


def test_custom_lexicon():
    n = ArabicNormalizer()
    n.add_lexicon({"تيوت": "تُوت"})
    assert "تُوت" in n("فويس تيوت")


# --------------------------------------------------------------- engine text logic
def test_language_resolution():
    assert resolve_language("ar") == "arz"
    assert resolve_language("English") == "en"
    with pytest.raises(ValueError):
        resolve_language("fr")


def test_sentence_split():
    chunks = split_sentences("جملة اولى؟ جملة تانية. جملة تالتة!")
    assert len(chunks) == 3


def test_long_sentence_wrap():
    long = "كلمة، " * 80
    chunks = split_sentences(long, max_chars=100)
    assert all(len(c) <= 110 for c in chunks)
