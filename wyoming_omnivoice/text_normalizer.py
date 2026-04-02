import logging
import re
from num2words import num2words
import eng_to_ipa as ipa

log = logging.getLogger(__name__)


class _EnglishToRussianNormalizer:
    """Class for converting English words to Russian phonetic representation."""
    
    SIMPLE_ENGLISH_TO_RUSSIAN = {
        'a': 'э', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
        'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
        'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
        'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'и', 'z': 'з'
    }

    # All keys in lowercase for reliable matching
    ENGLISH_EXCEPTIONS = {
        "google": "гугл", "apple": "эпл", "microsoft": "майкрософт",
        "samsung": "самсунг", "toyota": "тойота", "volkswagen": "фольцваген",
        "coca": "кока", "cola": "кола", "pepsi": "пэпси", "whatsapp": "вотсап",
        "telegram": "телеграм", "youtube": "ютуб", "instagram": "инстаграм",
        "facebook": "фэйсбук", "twitter": "твиттер", "iphone": "айфон",
        "tesla": "тесла", "spacex": "спэйс икс", "amazon": "амазон",
        "python": "пайтон", "ai": "эй ай", "api": "эйпиай",
        "it": "ай ти", "wi-fi": "вай фай", "wifi": "вай фай", "rtx": "эрте икс",
        "work": "ворк", "world": "ворлд", "bird": "бёрд",
        "girl": "гёрл", "burn": "бёрн", "her": "хёр",
        "early": "ёрли", "service": "сёрвис",
        "a": "э", "the": "зе", "of": "оф", "and": "энд", "for": "фо",
        "to": "ту", "in": "ин", "on": "он", "is": "из",
        "knowledge": "ноуледж", "new": "нью",
        "video": "видео", "ru": "ру", "com": "ком",
        "hot": "хот", "https": "аштитипиэс", "http": "аштитипи",
    }

    IPA_TO_RUSSIAN_MAP = {
        "ˈ": "", "ˌ": "", "ː": "",
        "p": "п", "b": "б", "t": "т", "d": "д", "k": "к", "g": "г",
        "m": "м", "n": "н", "f": "ф", "v": "в", "s": "с", "z": "з",
        "h": "х", "l": "л", "r": "р", "w": "в", "j": "й", "ʃ": "ш", 
        "ʒ": "ж", "tʃ": "ч", "ʧ": "ч", "dʒ": "дж", "ʤ": "дж", "ŋ": "нг", 
        "θ": "с", "ð": "з",
        "i": "и", "ɪ": "и", "ɛ": "э", "æ": "э", "ɑ": "а", "ɔ": "о",
        "u": "у", "ʊ": "у", "ʌ": "а", "ə": "э",
        "ər": "эр", "ɚ": "эр",
        "eɪ": "эй", "aɪ": "ай", "ɔɪ": "ой", "aʊ": "ау", "oʊ": "оу",
        "ɪə": "иэ", "eə": "еэ", "ʊə": "уэ",
    }

    def __init__(self):
        self._max_ipa_key_len = max(len(key) for key in self.IPA_TO_RUSSIAN_MAP.keys())

    def _convert_ipa_to_russian(self, ipa_text: str) -> str:
        result = ""
        pos = 0
        while pos < len(ipa_text):
            found_match = False
            for length in range(self._max_ipa_key_len, 0, -1):
                chunk = ipa_text[pos:pos + length]
                if chunk in self.IPA_TO_RUSSIAN_MAP:
                    result += self.IPA_TO_RUSSIAN_MAP[chunk]
                    pos += length
                    found_match = True
                    break
            if not found_match:
                pos += 1
        return result

    def _transliterate_word(self, match):
        word_original = match.group(0)
        word_lower = word_original.lower()
        
        # Check exceptions first
        if word_lower in self.ENGLISH_EXCEPTIONS:
            return self.ENGLISH_EXCEPTIONS[word_lower]

        try:
            # Get phonetic IPA
            ipa_transcription = ipa.convert(word_lower)
            ipa_transcription = re.sub(r'[/]', '', ipa_transcription).strip()
            
            # eng_to_ipa returns '*' at the end if it doesn't know the word
            if '*' in ipa_transcription: 
                raise ValueError("IPA conversion failed.")
                
            russian_phonetics = self._convert_ipa_to_russian(ipa_transcription)
            # Post-fix double sounds
            russian_phonetics = re.sub(r'йй', 'й', russian_phonetics)
            russian_phonetics = re.sub(r'([чшщждж])ь', r'\1', russian_phonetics)
            return russian_phonetics
        except Exception:
            # Fallback to character-by-character if IPA failed
            return ''.join(self.SIMPLE_ENGLISH_TO_RUSSIAN.get(c, c) for c in word_lower)

    def normalize(self, text: str) -> str:

        return re.sub(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b', self._transliterate_word, text)


class TextNormalizer:
    _emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\u2600-\u26FF" u"\u2700-\u27BF"
        u"\U0001F900-\U0001F9FF" u"\u200D" u"\uFE0F"
        "]+",
        flags=re.UNICODE
    )
    _chars_to_delete = "=#$“”„«»<>*\"‘’‚‹›'/"
    _map_from = "—–−\xa0"
    _map_to = "--- "
    _translation_table = str.maketrans(_map_from, _map_to, _chars_to_delete)
    
    _FINAL_CLEANUP_PATTERN = re.compile(r'[^а-яА-ЯёЁ?!., -]+')

    _OMNIVOICE_TAGS = [
        r"\[laughter\]", r"\[confirmation-en\]", r"\[question-en\]",
        r"\[question-ah\]", r"\[question-oh\]", r"\[question-ei\]",
        r"\[question-yi\]", r"\[surprise-ah\]", r"\[surprise-oh\]",
        r"\[surprise-wa\]", r"\[surprise-yo\]", r"\[dissatisfaction-hnn\]",
        r"\[sniff\]", r"\[sigh\]"
    ]
    _TAGS_PATTERN = re.compile(r'(' + '|'.join(_OMNIVOICE_TAGS) + r')')

    def __init__(self):
        self._eng_normalizer = _EnglishToRussianNormalizer()

    def normalize(self, text: str) -> str:
        # Isolate OmniVoice tags from normalization
        parts = self._TAGS_PATTERN.split(text)
        normalized_parts = []
        
        for part in parts:
            if not part:
                continue
            if self._TAGS_PATTERN.match(part):
                normalized_parts.append(part)
            else:
                normalized_parts.append(self._normalize_pipeline(part))
                
        result = " ".join(normalized_parts)
        return re.sub(r'\s+', ' ', result).strip()

    def _normalize_pipeline(self, text: str) -> str:
        if not text.strip():
            return text
            
        # 1. Percentages (100% -> 100 процентов)
        text = self._normalize_percentages(text)
        # 2. Basic symbols and emojis
        text = self._normalize_special_chars(text)
        # 3. Numeric plus (+7 -> плюс семь)
        text = self._normalize_plus_before_number(text)
        # 4. Numbers to words (100 -> сто)
        text = self._normalize_numbers(text)
        # 5. English to Russian phonetics (Wi-Fi -> вай фай)
        text = self._normalize_english(text)
        # 6. Final safety cleanup
        text = self._cleanup_final_text(text).strip()
        return text

    def _normalize_plus_before_number(self, text: str) -> str:
        return re.sub(r'\+(?=\d)', ' плюс ', text)

    def _cleanup_final_text(self, text: str) -> str:
        return self._FINAL_CLEANUP_PATTERN.sub(' ', text)

    def _choose_percent_form(self, number_str: str) -> str:
        if '.' in number_str or ',' in number_str: return "процента"
        try:
            number = int(number_str)
            if 10 < number % 100 < 20: return "процентов"
            last_digit = number % 10
            if last_digit == 1: return "процент"
            if last_digit in [2, 3, 4]: return "процента"
            return "процентов"
        except (ValueError, OverflowError): return "процентов"

    def _normalize_percentages(self, text: str) -> str:
        def replace_match(match):
            number_str_clean = match.group(1).replace(',', '.')
            percent_word = self._choose_percent_form(number_str_clean)
            return f" {number_str_clean} {percent_word} "
        return re.sub(r'(\d+([.,]\d+)?)\s*\%', replace_match, text)

    def _normalize_special_chars(self, text: str) -> str:
        text = self._emoji_pattern.sub(r'', text)
        text = text.translate(self._translation_table)
        text = text.replace('…', '.')
        text = re.sub(r':(?!\d)', ',', text)
        text = re.sub(r'([a-zA-Zа-яА-ЯёЁ])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Zа-яА-ЯёЁ])', r'\1 \2', text)
        text = text.replace('\n', ' ').replace('\t', ' ')
        return text

    def _normalize_numbers(self, text: str) -> str:
        def replace_number(match):
            num_str = match.group(0).replace(',', '.')
            try:
                if '.' in num_str:
                    parts = num_str.split('.')
                    integer_part_str, fractional_part_str = parts[0], parts[1]
                    if not integer_part_str or not fractional_part_str:
                        return num2words(int(num_str.replace('.', '')), lang='ru')
                    integer_words = num2words(int(integer_part_str), lang='ru')
                    fractional_words = num2words(int(fractional_part_str), lang='ru')
                    # Basic decimal handling
                    return f"{integer_words} и {fractional_words}"
                else: 
                    return num2words(int(num_str), lang='ru')
            except Exception:
                return num_str
        return re.sub(r'\b\d+([.,]\d+)?\b', replace_number, text)

    def _normalize_english(self, text: str) -> str:
        return self._eng_normalizer.normalize(text)