import re


class ClaimAnalyzer:

    CLAIM_THRESHOLD = 0.50

    def __init__(self):

        print("Claim Analyzer loaded.")

    # =====================================================
    # CLEAN TEXT
    # =====================================================

    def clean_text(self, text):

        if not text:
            return ""

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # =====================================================
    # TOKENIZE FOR EXACT MATCHING
    #
    # Example:
    #
    # "විය" matches "විය"
    # but NOT "වියදම්"
    # =====================================================

    def get_tokens(self, text):

        if not text:
            return []

        return re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?"
            r"|[\u0D80-\u0DFF]+"
            r"|\d+(?:\.\d+)?%?",
            text.lower()
        )

    # =====================================================
    # CHECK EXACT WORD / PHRASE
    # =====================================================

    def contains_marker(
        self,
        sentence,
        marker
    ):

        sentence_tokens = self.get_tokens(
            sentence
        )

        marker_tokens = self.get_tokens(
            marker
        )

        if not marker_tokens:
            return False

        marker_length = len(
            marker_tokens
        )

        for i in range(
            len(sentence_tokens)
            - marker_length
            + 1
        ):

            if (
                sentence_tokens[
                    i:i + marker_length
                ]
                == marker_tokens
            ):

                return True

        return False

    # =====================================================
    # CHECK WHETHER TEXT CONTAINS SINHALA
    # =====================================================

    def contains_sinhala_text(
        self,
        text
    ):

        if not text:
            return False

        return bool(
            re.search(
                r"[\u0D80-\u0DFF]",
                text
            )
        )

    # =====================================================
    # SPLIT TRANSCRIPT INTO SENTENCES / CLAUSES
    #
    # English:
    # mainly sentence punctuation
    #
    # Sinhala:
    # also split long spoken-ASR text using commas
    #
    # Example:
    #
    # ගොවියොන්ට පෝර ටික නෑ,
    # රජයේ සේවකයෝ අමාරුවෙන් ජීවත් වෙන්නේ,
    #
    # becomes two separate claim candidates.
    # =====================================================

    def split_sinhala_clauses(self, sentence):
        """Split only after a predicate and before a known new subject.

        Source spans remain verbatim. This is a conservative heuristic,
        not a Sinhala parser or factuality check.
        """
        # Preserve reporting/conditional scope and question context.
        # If scope is uncertain, leave the sentence intact.
        protected = (
            "කියලා", "කියා", "කියන", "බව", "බවට", "නම්", "නිසා",
            "නමුත්", "හැබැයි", "එහෙත්", "වුණත්", "වුනත්", "නේද",
        )
        if self.is_question(sentence) or any(
            self.contains_marker(sentence, marker) for marker in protected
        ):
            return [sentence]

        endings = (
            "ඉන්නවා", "සිටිනවා", "තියෙනවා", "තිබෙනවා", "තිබුණා",
            "වෙනවා", "වෙන්නේ", "වුණා", "කළා", "බෑ", "නෑ", "නැහැ",
            "නොවේ", "නොවෙයි", "නොහැක", "කරලා",
        )
        letters = r"[\u0D80-\u0DFF\u200c\u200d]"
        predicate = re.compile(
            r"(?<!" + letters + r")(?:" + "|".join(map(re.escape, endings))
            + r")(?!" + letters + r")"
        )
        # Extend this vocabulary only with reviewed examples.
        subject = re.compile(
            r"^(?:ගොවියොන්ට|ගොවියන්ට|ගොවියෝ|ගොවියන්|"
            r"රජයේ\s+සේවකයෝ|පොලිස්\s+සේවාව|"
            r"සෞඛ්‍ය\s+සේවාව|රෝහල්|ආණ්ඩුව|රජය|ජනතාව)"
            r"(?=$|[\s,，;.!?])"
        )
        clauses = []
        start = 0
        for match in predicate.finditer(sentence):
            separator = re.match(r"[\s,，;]+", sentence[match.end():])
            if not separator:
                continue
            next_start = match.end() + separator.end()
            if not subject.match(sentence[next_start:]):
                continue
            # 'කරලා' can introduce a continuation. Require punctuation
            # as well as a new subject before treating it as a boundary.
            if match.group() == "කරලා" and not any(
                char in separator.group() for char in ",，;"
            ):
                continue
            clauses.append(sentence[start:match.end()].strip())
            start = next_start
        remainder = sentence[start:].strip()
        if remainder:
            clauses.append(remainder)
        return clauses

    def split_sentences(self, text, language=None):
        text = self.clean_text(text)
        if not text:
            return []
        # Retain the existing English sentence boundaries.
        base_sentences = re.split(r"(?<=[.!?])\s+|[;\n]+", text)
        is_sinhala = language == "si" or self.contains_sinhala_text(text)
        final_sentences = []
        for sentence in base_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            clauses = self.split_sinhala_clauses(sentence) if is_sinhala else [sentence]
            final_sentences.extend(c.strip(" ,;:") for c in clauses if c.strip(" ,;:"))
        return final_sentences

    # =====================================================
    # CHECK QUESTION
    # =====================================================

    def is_question(
        self,
        sentence
    ):

        if not sentence:
            return False

        if sentence.strip().endswith("?"):
            return True

        question_markers = [

            # English
            "what",
            "why",
            "when",
            "where",
            "who",
            "how",

            # Sinhala
            "මොකක්",
            "මොනවද",
            "ඇයි",
            "කවදා",
            "කොහෙද",
            "කවුද",
            "කොහොමද"
        ]

        return any(

            self.contains_marker(
                sentence,
                marker
            )

            for marker
            in question_markers
        )

    # =====================================================
    # CHECK OPINION / ADVICE
    # =====================================================

    def contains_advice_or_opinion(
        self,
        sentence
    ):

        markers = [

            # English
            "i think",
            "i believe",
            "in my opinion",
            "you should",
            "should",
            "recommend",
            "must",
            "you need",
            "you need to",

            # Sinhala
            "මම හිතන්නේ",
            "මගේ අදහස",
            "හිතන්න",
            "කරන්න",
            "අයින් කරන්න",
            "ගන්න ඕන",
            "වෙන්න ඕන",
            "කරන්න ඕන",
            "යන්න ඕන",
            "යුතුයි",
            "හොඳයි",
            "පටන් ගන්න",
            "මම කිව්වා",

            # Additional conversational advice
            "ඉතුරු කරන්න",
            "තියාගන්න",
            "තියා ගන්න",
            "බලන්න",
            "හිතලා බලන්න"
        ]

        return any(

            self.contains_marker(
                sentence,
                marker
            )

            for marker
            in markers
        )

    # =====================================================
    # CHECK PERSONAL / SUBJECTIVE LANGUAGE
    # =====================================================

    def contains_personal_signal(
        self,
        sentence
    ):

        markers = [

            # English
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",

            # Sinhala
            "මම",
            "මගේ",
            "අපි",
            "අපේ",
            "ඔයා",
            "ඔයාගේ",
            "ඔබ",
            "ඔබගේ"
        ]

        return any(

            self.contains_marker(
                sentence,
                marker
            )

            for marker
            in markers
        )

    # =====================================================
    # CHECK NUMERIC INFORMATION
    # =====================================================

    def contains_number(
        self,
        sentence
    ):

        return bool(
            re.search(
                r"\d+(?:\.\d+)?%?",
                sentence
            )
        )

    # =====================================================
    # CHECK POSSIBLE ENGLISH NAMED ENTITY
    #
    # Temporary heuristic.
    #
    # Real NER runs in the next pipeline stage.
    # =====================================================

    def contains_proper_name_signal(
        self,
        sentence
    ):

        english_words = re.findall(
            r"[A-Za-z]+",
            sentence
        )

        if not english_words:
            return False

        capitalized_words = 0
        acronyms = 0

        for index, word in enumerate(
            english_words
        ):

            if len(word) < 2:
                continue

            # Acronyms:
            # CID, BBC, WHO

            if (
                word.isupper()
                and len(word) >= 2
            ):

                acronyms += 1
                continue

            # Ignore ordinary first word
            # capitalization.

            if (
                index == 0
                and word[0].isupper()
                and word[1:].islower()
            ):

                continue

            if (
                word[0].isupper()
                and word[1:].islower()
            ):

                capitalized_words += 1

        # Example:
        # Suresh Saleh
        # Colombo National Hospital

        if capitalized_words >= 2:
            return True

        # Example:
        # CID

        if acronyms >= 1:
            return True

        return False

    # =====================================================
    # STRONG FACTUAL / EVENT SIGNAL
    #
    # Formal English + Sinhala factual language.
    #
    # These are given stronger scoring weight.
    # =====================================================

    def contains_factual_signal(
        self,
        sentence
    ):

        factual_markers = [

            # English
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "reported",
            "announced",
            "confirmed",
            "arrested",
            "admitted",
            "released",
            "killed",
            "injured",
            "occurred",
            "happened",
            "held",
            "detained",
            "remains",
            "died",
            "charged",
            "visited",
            "celebrated",
            "observed",

            # Sinhala - formal news language
            "තිබේ",
            "සිටී",
            "විය",
            "වී ඇත",
            "සිදු විය",
            "සිදුවී ඇත",
            "අත්අඩංගුවට",
            "අත්අඩංගුවට ගෙන ඇත",
            "රෝහල්ගත",
            "රෝහල්ගත කර ඇත",
            "නිවේදනය කළේය",
            "නිවේදනය කර ඇත",
            "තහවුරු කළේය",
            "තහවුරු කර ඇත",
            "මිය ගියේය",
            "මියගොස් ඇත",
            "තුවාල ලබා ඇත",
            "වාර්තා වී ඇත",
            "පැවැත්විණි"
        ]

        return any(

            self.contains_marker(
                sentence,
                marker
            )

            for marker
            in factual_markers
        )

    # =====================================================
    # NATURAL / SPOKEN SINHALA FACTUAL SIGNAL
    #
    # Real Sinhala videos often use conversational
    # language instead of formal written-news language.
    #
    # Examples:
    #
    # ගොවියොන්ට පෝර ටික නෑ
    # පොලිස් සේවාව විනාශ කරලා
    # සේවකයෝ අමාරුවෙන් ජීවත් වෙන්නේ
    #
    # This is intentionally scored more carefully
    # than strong formal factual markers.
    # =====================================================

    def contains_spoken_sinhala_factual_signal(
        self,
        sentence
    ):

        if not self.contains_sinhala_text(
            sentence
        ):

            return False

        markers = [

            "නෑ",
            "නැහැ",
            "බෑ",

            "තියෙනවා",
            "තිබෙනවා",
            "තිබුණා",

            "ඉන්නවා",
            "සිටිනවා",
            "සිටිනවා",

            "වෙනවා",
            "වෙන්නේ",
            "වුණා",

            "කරලා",
            "කරල",
            "කරමින් ඉන්නවා",
            "කරමින් සිටිනවා",

            "සිදු වෙලා",
            "සිදුවෙලා",

            "වාර්තා වෙනවා",
            "වාර්තා වෙලා",

            "පැවැත්වුණා",
            "පැවැත්වෙනවා",

            "ලැබුණා",
            "ලැබෙනවා"
        ]

        return any(

            self.contains_marker(
                sentence,
                marker
            )

            for marker
            in markers
        )

    # =====================================================
    # PUBLIC / INSTITUTION / SOCIAL SUBJECT SIGNAL
    #
    # Helps distinguish a check-worthy public claim
    # from ordinary personal conversation.
    #
    # Examples:
    #
    # ආණ්ඩුව
    # රජය
    # පොලිසිය
    # ගොවියෝ
    # සෞඛ්‍ය සේවාව
    # අධ්‍යාපනය
    # =====================================================

    def contains_sinhala_public_subject_signal(
        self,
        sentence
    ):

        if not self.contains_sinhala_text(
            sentence
        ):

            return False

        markers = [

            "ආණ්ඩුව",
            "රජය",
            "රජයේ",

            "පොලිසිය",
            "පොලිස්",

            "හමුදාව",

            "ජනතාව",

            "ගොවියෝ",
            "ගොවියන්",
            "ගොවියොන්ට",

            "රජයේ සේවකයෝ",
            "සේවකයෝ",

            "සෞඛ්‍ය",
            "සෞඛ්‍ය සේවාව",

            "අධ්‍යාපනය",

            "රෝහල",
            "රෝහල්",

            "පාසල්",
            "විශ්වවිද්‍යාල",

            "රට",
            "රටේ",

            "සේවාව",
            "පද්ධතිය",
            "සමාජය",

            "අමාත්‍යාංශය",
            "අමාත්‍යංශය",

            "පාර්ලිමේන්තුව",

            "ජනාධිපති",
            "අගමැති",
            "අමාත්‍ය"
        ]

        return any(

            self.contains_marker(
                sentence,
                marker
            )

            for marker
            in markers
        )

    # =====================================================
    # TIME SIGNAL
    # =====================================================

    def contains_time_signal(
        self,
        sentence
    ):

        time_markers = [

            # English
            "today",
            "yesterday",
            "tomorrow",

            "year",
            "years",

            "month",
            "months",

            "week",
            "weeks",

            "day",
            "days",

            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",

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
            "december",

            "this year",
            "last year",
            "next year",

            # Sinhala
            "අද",
            "ඊයේ",
            "හෙට",

            "වසර",
            "වසරේ",

            "අවුරුදු",
            "අවුරුද්දේ",

            "මාස",
            "මාසය",
            "මාසයේ",

            "සතිය",
            "සතියේ",

            "දින",
            "දිනය",

            "මේ වසරේ",
            "පසුගිය වසරේ",
            "ලබන වසරේ",

            "මේ අවුරුද්දේ",
            "පසුගිය අවුරුද්දේ",
            "ලබන අවුරුද්දේ",

            "සඳුදා",
            "අඟහරුවාදා",
            "බදාදා",
            "බ්‍රහස්පතින්දා",
            "සිකුරාදා",
            "සෙනසුරාදා",
            "ඉරිදා"
        ]

        if any(

            self.contains_marker(
                sentence,
                marker
            )

            for marker
            in time_markers
        ):

            return True

        # Year:
        # 2019, 2026

        if re.search(
            r"\b(?:19|20)\d{2}\b",
            sentence
        ):

            return True

        # Numeric date:
        # 10/10/2026
        # 10-10-2026

        if re.search(
            r"\b\d{1,2}[/-]\d{1,2}"
            r"[/-]\d{2,4}\b",
            sentence
        ):

            return True

        return False

    # =====================================================
    # SCORE ONE SENTENCE / CLAUSE
    #
    # IMPORTANT:
    #
    # This score means CHECK-WORTHINESS.
    #
    # It does NOT mean:
    #
    # probability that the claim is true
    # probability that the claim is fake
    # =====================================================

    def calculate_claim_score(
        self,
        sentence
    ):

        words = sentence.split()

        # Too short / likely fragment.

        if len(words) < 4:
            return 0.0

        # Questions are not factual claims.

        if self.is_question(
            sentence
        ):

            return 0.0

        score = 0.0

        # ---------------------------------------------
        # Basic declarative statement
        # ---------------------------------------------

        score += 0.20

        # ---------------------------------------------
        # Numeric information
        # ---------------------------------------------

        if self.contains_number(
            sentence
        ):

            score += 0.20

        # ---------------------------------------------
        # Possible English named entity
        # ---------------------------------------------

        if self.contains_proper_name_signal(
            sentence
        ):

            score += 0.20

        # ---------------------------------------------
        # Strong formal factual / event signal
        # ---------------------------------------------

        strong_factual = (
            self.contains_factual_signal(
                sentence
            )
        )

        spoken_sinhala = (
            self.contains_spoken_sinhala_factual_signal(
                sentence
            )
        )

        public_subject = (
            self.contains_sinhala_public_subject_signal(
                sentence
            )
        )

        if strong_factual:

            score += 0.35

        # ---------------------------------------------
        # Natural spoken Sinhala
        #
        # We deliberately do NOT give conversational
        # expressions the same weight as formal factual
        # markers.
        #
        # Spoken factual marker = +0.20
        #
        # Public/institution subject = extra +0.20
        # ---------------------------------------------

        elif spoken_sinhala:

            score += 0.20

            if public_subject:

                score += 0.20

        # ---------------------------------------------
        # Time / date information
        # ---------------------------------------------

        if self.contains_time_signal(
            sentence
        ):

            score += 0.15

        # ---------------------------------------------
        # Advice / opinion penalty
        # ---------------------------------------------

        if self.contains_advice_or_opinion(
            sentence
        ):

            score -= 0.30

        # ---------------------------------------------
        # Personal-language penalty
        # ---------------------------------------------

        if self.contains_personal_signal(
            sentence
        ):

            score -= 0.10

        # ---------------------------------------------
        # Keep score between 0 and 1
        # ---------------------------------------------

        score = max(
            0.0,
            min(
                score,
                1.0
            )
        )

        return round(
            score,
            3
        )

    # =====================================================
    # EXPLAIN WHY SENTENCE WAS SELECTED
    # =====================================================

    def get_reasons(
        self,
        sentence
    ):

        reasons = []

        if self.contains_number(
            sentence
        ):

            reasons.append(
                "contains numeric information"
            )

        if self.contains_proper_name_signal(
            sentence
        ):

            reasons.append(
                "contains possible named entity"
            )

        if self.contains_factual_signal(
            sentence
        ):

            reasons.append(
                "contains factual/event language"
            )

        if self.contains_spoken_sinhala_factual_signal(
            sentence
        ):

            reasons.append(
                "contains natural spoken Sinhala factual language"
            )

        if self.contains_sinhala_public_subject_signal(
            sentence
        ):

            reasons.append(
                "contains public/institution subject"
            )

        if self.contains_time_signal(
            sentence
        ):

            reasons.append(
                "contains time-related information"
            )

        if self.contains_advice_or_opinion(
            sentence
        ):

            reasons.append(
                "contains advice/opinion language"
            )

        if self.contains_personal_signal(
            sentence
        ):

            reasons.append(
                "contains personal-language signal"
            )

        if not reasons:

            reasons.append(
                "declarative statement"
            )

        return reasons

    # =====================================================
    # MAIN CLAIM ANALYSIS
    # =====================================================

    def get_review_reasons(self, sentence):
        """Conservative Sinhala review cues; never a truth assessment.

        An empty list means no configured cue matched, not that the
        statement is verified, clear, or ready for evidence retrieval.
        """
        if not self.contains_sinhala_text(sentence):
            return []
        reasons = []
        broad_wording = (
            "විනාස", "විනාශ", "අමාරුවෙන්", "නැති කරමින්",
        )
        if any(self.contains_marker(sentence, m) for m in broad_wording):
            reasons.append("broad or evaluative wording; clarify the specific allegation")

        # Multiple domains can form a valid single claim; flag for review,
        # rather than assuming they must be split or removing the claim.
        topic_groups = (
            ("සෞඛ්‍ය", "රෝහල", "රෝහල්"),
            ("ගොවියෝ", "ගොවියන්", "ගොවියොන්ට", "ගොවියන්ට", "පෝර", "වී"),
            ("රජයේ සේවකයෝ",),
            ("පොලිස්", "පොලිසිය"),
            ("අධ්‍යාපනය", "තරුණ පරපුර"),
            ("සංස්කෘතිය",),
            ("සංඝ",),
        )
        topic_count = sum(
            any(self.contains_marker(sentence, m) for m in group)
            for group in topic_groups
        )
        if topic_count > 1:
            reasons.append("multiple topic cues; check whether this is one claim")
        if len(sentence.split()) > 25:
            reasons.append("long statement; check scope and attached qualifications")
        return reasons

    def analyze(
        self,
        transcript,
        language=None
    ):

        transcript = self.clean_text(
            transcript
        )

        if not transcript:

            return {
                "language": language,
                "sentence_count": 0,
                "claim_count": 0,
                "claim_threshold":
                    self.CLAIM_THRESHOLD,
                "claims": []
            }

        # IMPORTANT:
        # Language is now passed into sentence/clause
        # splitting.

        sentences = self.split_sentences(
            transcript,
            language=language
        )

        claims = []

        claim_id = 1

        for sentence in sentences:

            score = (
                self.calculate_claim_score(
                    sentence
                )
            )

            # Candidate claim threshold

            if score >= self.CLAIM_THRESHOLD:

                claim = {

                    "claim_id":
                        claim_id,

                    "text":
                        sentence,

                    "check_worthy_score":
                        score,

                    "reasons":
                        self.get_reasons(
                            sentence
                        )
                }

                review_reasons = self.get_review_reasons(sentence)
                claim["needs_review"] = bool(review_reasons)
                claim["review_reasons"] = review_reasons
                # This records limited rule screening, never approval or truth.
                screened = (language in (None, "si")
                            and self.contains_sinhala_text(sentence))
                claim["review_screening_status"] = (
                    "heuristic_screened" if screened else "not_performed"
                )
                claim["review_screening_method"] = (
                    "sinhala_rules_v1" if screened else None
                )

                claims.append(
                    claim
                )

                claim_id += 1

        return {
            "language":
                language,

            "sentence_count":
                len(sentences),

            "claim_count":
                len(claims),

            "claim_threshold":
                self.CLAIM_THRESHOLD,

            "claims":
                claims
        }