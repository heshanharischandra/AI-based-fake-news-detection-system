import re
import unicodedata

from transformers import pipeline


class NERExtractor:

    MODEL_NAME = (
        "asanka25/"
        "xlm-roberta-base-finetuned-conll03-english-finetuned-sinhala"
    )

    def __init__(self):

        print("Loading NER model...")

        self.ner_pipeline = pipeline(
            "token-classification",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            aggregation_strategy="simple",
            device=-1
        )

        print("NER model loaded successfully.")

    # =====================================================
    # CLEAN ENTITY TEXT
    # =====================================================

    def clean_entity_text(self, text):

        if not text:
            return ""

        # Normalize Unicode representation.
        text = unicodedata.normalize(
            "NFC",
            text
        )

        # XLM-R sometimes uses ▁ for word boundaries.
        text = text.replace(
            "▁",
            " "
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # =====================================================
    # ADD UNIQUE VALUE
    # =====================================================

    def add_unique(
        self,
        collection,
        value
    ):

        value = self.clean_entity_text(
            value
        )

        if not value:
            return

        if value not in collection:
            collection.append(value)

    # =====================================================
    # NORMALIZE LOCATION
    #
    # Examples:
    #
    # කොළඹදී
    #     ↓
    # කොළඹ
    #
    # ශ්‍රී ලංකාවේ
    #     ↓
    # ශ්‍රී ලංකාව
    # =====================================================

    def normalize_location(
        self,
        text
    ):

        text = self.clean_entity_text(
            text
        )

        if not text:
            return ""

        # -------------------------------------------------
        # Create a comparison-only version without
        # hidden zero-width Sinhala Unicode characters.
        #
        # We do NOT use this version as the final output.
        # -------------------------------------------------

        comparison_text = (
            text
            .replace("\u200d", "")
            .replace("\u200c", "")
        )

        # -------------------------------------------------
        # NORMALIZE SRI LANKA
        #
        # Handles:
        #
        # ශ්‍රී ලංකාවේ
        # ශ්‍රී ලංකා
        # ශ්‍රී ලංකාව
        #
        # even when a hidden ZWJ exists inside "ශ්‍රී".
        # -------------------------------------------------

        if (
            "ලංකාවේ" in comparison_text
            or comparison_text.endswith(
                "ශ්රී ලංකා"
            )
            or comparison_text.endswith(
                "ශ්රී ලංකාව"
            )
        ):

            return "ශ්‍රී ලංකාව"

        # -------------------------------------------------
        # SINHALA LOCATION SUFFIX
        #
        # කොළඹදී -> කොළඹ
        # -------------------------------------------------

        if (
            text.endswith("දී")
            and len(text) > 2
        ):

            text = text[:-2]

        return text.strip()

    # =====================================================
    # DATE / TIME EXPRESSION EXTRACTION
    #
    # Supports:
    #
    # Exact dates
    # Years
    # Months
    # Weekdays
    # Relative time expressions
    #
    # English + Sinhala
    # =====================================================

    def extract_dates(self, text):

        dates = []

        if not text:
            return dates

        # =================================================
        # 1. NUMERIC DATES
        #
        # 10/10/2026
        # 10-10-2026
        # =================================================

        numeric_dates = re.findall(
            r"\b\d{1,2}[/-]\d{1,2}"
            r"[/-]\d{2,4}\b",
            text
        )

        for date in numeric_dates:

            self.add_unique(
                dates,
                date
            )

        # =================================================
        # 2. ENGLISH MONTHS
        # =================================================

        english_months = (
            r"January|February|March|April|May|June|"
            r"July|August|September|October|November|December"
        )

        english_date_patterns = [

            # January 10, 2026
            rf"\b(?:{english_months})"
            rf"\s+\d{{1,2}}(?:,\s*|\s+)"
            rf"(?:19|20)\d{{2}}\b",

            # 10 January 2026
            rf"\b\d{{1,2}}\s+"
            rf"(?:{english_months})\s+"
            rf"(?:19|20)\d{{2}}\b"
        ]

        for pattern in english_date_patterns:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for date in matches:

                self.add_unique(
                    dates,
                    date
                )

        # =================================================
        # 3. SINHALA MONTHS
        # =================================================

        sinhala_months = (
            r"ජනවාරි|පෙබරවාරි|මාර්තු|අප්‍රේල්|"
            r"මැයි|ජූනි|ජූලි|අගෝස්තු|"
            r"සැප්තැම්බර්|ඔක්තෝබර්|නොවැම්බර්|දෙසැම්බර්"
        )

        sinhala_date_patterns = [

            # 2026 ඔක්තෝබර් 10
            rf"\b(?:19|20)\d{{2}}\s+"
            rf"(?:{sinhala_months})\s+"
            rf"\d{{1,2}}",

            # 10 ඔක්තෝබර් 2026
            rf"\b\d{{1,2}}\s+"
            rf"(?:{sinhala_months})\s+"
            rf"(?:19|20)\d{{2}}"
        ]

        for pattern in sinhala_date_patterns:

            matches = re.findall(
                pattern,
                text
            )

            for date in matches:

                self.add_unique(
                    dates,
                    date
                )

        # =================================================
        # 4. ENGLISH WEEKDAYS
        # =================================================

        english_weekdays = re.findall(
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|"
            r"Friday|Saturday|Sunday)\b",
            text,
            flags=re.IGNORECASE
        )

        for weekday in english_weekdays:

            self.add_unique(
                dates,
                weekday
            )

        # =================================================
        # 5. SINHALA WEEKDAYS
        # =================================================

        sinhala_weekday_pattern = (
            r"සඳුදා|"
            r"අඟහරුවාදා|"
            r"බදාදා|"
            r"බ්‍රහස්පතින්දා|"
            r"සිකුරාදා|"
            r"සෙනසුරාදා|"
            r"ඉරිදා"
        )

        sinhala_weekdays = re.findall(
            sinhala_weekday_pattern,
            text
        )

        for weekday in sinhala_weekdays:

            self.add_unique(
                dates,
                weekday
            )

        # =================================================
        # 6. ENGLISH RELATIVE TIME EXPRESSIONS
        # =================================================

        english_relative_patterns = [

            r"\btoday\b",
            r"\byesterday\b",
            r"\btomorrow\b",

            r"\bthis year\b",
            r"\blast year\b",
            r"\bnext year\b",

            r"\bthis month\b",
            r"\blast month\b",
            r"\bnext month\b",

            r"\bthis week\b",
            r"\blast week\b",
            r"\bnext week\b",

            r"\bthis weekend\b",
            r"\blast weekend\b",
            r"\bnext weekend\b"
        ]

        for pattern in english_relative_patterns:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for expression in matches:

                self.add_unique(
                    dates,
                    expression
                )

        # =================================================
        # 7. SINHALA RELATIVE TIME EXPRESSIONS
        # =================================================

        sinhala_relative_expressions = [

            "අද",
            "ඊයේ",
            "හෙට",

            "මේ වසරේ",
            "පසුගිය වසරේ",
            "ලබන වසරේ",

            "මේ අවුරුද්දේ",
            "පසුගිය අවුරුද්දේ",
            "ලබන අවුරුද්දේ",

            "මේ මාසයේ",
            "පසුගිය මාසයේ",
            "ලබන මාසයේ",

            "මේ සතියේ",
            "පසුගිය සතියේ",
            "ලබන සතියේ"
        ]

        for expression in sinhala_relative_expressions:

            if expression in text:

                self.add_unique(
                    dates,
                    expression
                )

        # =================================================
        # 8. STANDALONE ENGLISH MONTHS
        # =================================================

        standalone_english_months = re.findall(
            rf"\b(?:{english_months})\b",
            text,
            flags=re.IGNORECASE
        )

        for month in standalone_english_months:

            already_in_date = any(
                month.lower() in date.lower()
                for date in dates
            )

            if not already_in_date:

                self.add_unique(
                    dates,
                    month
                )

        # =================================================
        # 9. STANDALONE SINHALA MONTHS
        # =================================================

        standalone_sinhala_months = re.findall(
            sinhala_months,
            text
        )

        for month in standalone_sinhala_months:

            already_in_date = any(
                month in date
                for date in dates
            )

            if not already_in_date:

                self.add_unique(
                    dates,
                    month
                )

        # =================================================
        # 10. STANDALONE YEARS
        # =================================================

        years = re.findall(
            r"\b(?:19|20)\d{2}\b",
            text
        )

        for year in years:

            already_in_date = any(
                year in date
                for date in dates
            )

            if not already_in_date:

                self.add_unique(
                    dates,
                    year
                )

        return dates

    # =====================================================
    # NUMBER EXTRACTION
    # =====================================================

    def extract_numbers(self, text):

        if not text:
            return []

        numbers = re.findall(
            r"\b\d+(?:\.\d+)?%?\b",
            text
        )

        unique_numbers = []

        for number in numbers:

            self.add_unique(
                unique_numbers,
                number
            )

        return unique_numbers

    # =====================================================
    # CHECK WHETHER MISC VALUE IS REALLY
    # DATE / NUMBER / TIME NOISE
    # =====================================================

    def is_date_or_number_misc(
        self,
        text
    ):

        if not text:
            return True

        cleaned = self.clean_entity_text(
            text
        )

        # Pure number / punctuation

        if re.fullmatch(
            r"[\d\s.,/%\-]+",
            cleaned
        ):

            return True

        # -------------------------------------------------
        # Sinhala month names
        # -------------------------------------------------

        sinhala_months = [

            "ජනවාරි",
            "පෙබරවාරි",
            "මාර්තු",
            "අප්‍රේල්",
            "මැයි",
            "ජූනි",
            "ජූලි",
            "අගෝස්තු",
            "සැප්තැම්බර්",
            "ඔක්තෝබර්",
            "නොවැම්බර්",
            "දෙසැම්බර්"
        ]

        # -------------------------------------------------
        # English month names
        # -------------------------------------------------

        english_months = [

            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december"
        ]

        # -------------------------------------------------
        # English weekdays
        # -------------------------------------------------

        english_weekdays = [

            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday"
        ]

        # -------------------------------------------------
        # Sinhala weekdays
        # -------------------------------------------------

        sinhala_weekdays = [

            "සඳුදා",
            "අඟහරුවාදා",
            "බදාදා",
            "බ්‍රහස්පතින්දා",
            "සිකුරාදා",
            "සෙනසුරාදා",
            "ඉරිදා"
        ]

        cleaned_lower = cleaned.lower()

        # Sinhala date fragments

        if (
            any(
                month in cleaned
                for month in sinhala_months
            )
            and re.search(
                r"\d",
                cleaned
            )
        ):

            return True

        # English date fragments

        if (
            any(
                month in cleaned_lower
                for month in english_months
            )
            and re.search(
                r"\d",
                cleaned
            )
        ):

            return True

        # Standalone English month

        if cleaned_lower in english_months:

            return True

        # Standalone Sinhala month

        if cleaned in sinhala_months:

            return True

        # English weekday

        if cleaned_lower in english_weekdays:

            return True

        # Sinhala weekday

        if cleaned in sinhala_weekdays:

            return True

        return False

    # =====================================================
    # RUN NER MODEL
    #
    # Character positions are used to reconstruct
    # split XLM-R entity pieces.
    # =====================================================

    def extract_model_entities(
        self,
        text
    ):

        result = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "miscellaneous": []
        }

        if not text:
            return result

        predictions = self.ner_pipeline(
            text
        )

        # =================================================
        # STEP 1 - PREPARE ENTITY PIECES
        # =================================================

        prepared = []

        for prediction in predictions:

            entity_group = prediction.get(
                "entity_group",
                prediction.get(
                    "entity",
                    ""
                )
            )

            entity_group = (
                entity_group
                .replace("B-", "")
                .replace("I-", "")
                .upper()
            )

            start = prediction.get(
                "start"
            )

            end = prediction.get(
                "end"
            )

            if (
                start is None
                or end is None
            ):

                continue

            prepared.append({
                "type": entity_group,
                "start": int(start),
                "end": int(end)
            })

        if not prepared:
            return result

        prepared = sorted(
            prepared,
            key=lambda item: (
                item["start"],
                item["end"]
            )
        )

        # =================================================
        # STEP 2 - MERGE SPLIT ENTITY PIECES
        # =================================================

        merged = []

        for item in prepared:

            if not merged:

                merged.append(
                    item.copy()
                )

                continue

            previous = merged[-1]

            gap = (
                item["start"]
                - previous["end"]
            )

            if (
                item["type"]
                == previous["type"]
                and gap <= 1
            ):

                previous["end"] = max(
                    previous["end"],
                    item["end"]
                )

            else:

                merged.append(
                    item.copy()
                )

        # =================================================
        # STEP 3 - GET ORIGINAL ENTITY TEXT
        # =================================================

        for entity in merged:

            entity_group = (
                entity["type"]
            )

            entity_text = text[
                entity["start"]:
                entity["end"]
            ]

            entity_text = (
                self.clean_entity_text(
                    entity_text
                )
            )

            if not entity_text:
                continue

            # ---------------------------------------------
            # PERSON
            # ---------------------------------------------

            if entity_group == "PER":

                self.add_unique(
                    result["persons"],
                    entity_text
                )

            # ---------------------------------------------
            # ORGANIZATION
            # ---------------------------------------------

            elif entity_group == "ORG":

                self.add_unique(
                    result["organizations"],
                    entity_text
                )

            # ---------------------------------------------
            # LOCATION
            # ---------------------------------------------

            elif entity_group == "LOC":

                normalized_location = (
                    self.normalize_location(
                        entity_text
                    )
                )

                self.add_unique(
                    result["locations"],
                    normalized_location
                )

            # ---------------------------------------------
            # MISCELLANEOUS
            # ---------------------------------------------

            elif entity_group == "MISC":

                # Ignore dates, numbers,
                # months and weekdays because
                # those are extracted separately.

                if self.is_date_or_number_misc(
                    entity_text
                ):

                    continue

                self.add_unique(
                    result["miscellaneous"],
                    entity_text
                )

        return result

    # =====================================================
    # EXTRACT ENTITIES FROM ONE CLAIM
    # =====================================================

    def extract(
        self,
        text,
        language=None
    ):

        if not text:

            return {
                "language": language,
                "persons": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "numbers": [],
                "miscellaneous": []
            }

        text = text.strip()

        if not text:

            return {
                "language": language,
                "persons": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "numbers": [],
                "miscellaneous": []
            }

        # ---------------------------------------------
        # MODEL ENTITIES
        # ---------------------------------------------

        model_entities = (
            self.extract_model_entities(
                text
            )
        )

        # ---------------------------------------------
        # DATE / TIME EXPRESSIONS
        # ---------------------------------------------

        dates = self.extract_dates(
            text
        )

        # ---------------------------------------------
        # NUMBERS
        # ---------------------------------------------

        numbers = self.extract_numbers(
            text
        )

        return {
            "language":
                language,

            "persons":
                model_entities[
                    "persons"
                ],

            "organizations":
                model_entities[
                    "organizations"
                ],

            "locations":
                model_entities[
                    "locations"
                ],

            "dates":
                dates,

            "numbers":
                numbers,

            "miscellaneous":
                model_entities[
                    "miscellaneous"
                ]
        }

    # =====================================================
    # ENRICH ALL DETECTED CLAIMS
    # =====================================================

    def enrich_claims(
        self,
        claims,
        language=None
    ):

        enriched_claims = []

        for claim in claims:

            claim_text = claim.get(
                "text",
                ""
            )

            entities = self.extract(
                claim_text,
                language=language
            )

            # Keep the original claim fields:
            #
            # claim_id
            # text
            # check_worthy_score
            # reasons

            enriched_claim = (
                claim.copy()
            )

            enriched_claim[
                "entities"
            ] = entities

            enriched_claims.append(
                enriched_claim
            )

        return enriched_claims