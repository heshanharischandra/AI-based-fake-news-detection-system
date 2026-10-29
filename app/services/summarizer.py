import re
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
class VideoSummarizer:
    """Language-specific summarization with an inspectable raw candidate.
    Lexical checks are heuristics, not factual verification. AI acceptance
    Sinhala AI acceptance is disabled. English DistilBART candidates use
    conservative heuristic screening, which cannot guarantee factuality.
    """
    MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
    ENGLISH_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
    INPUT_LIMIT = 512
    MAX_SUMMARY_SENTENCES = 3
    SUMMARY_WORD_BUDGET = 100
    SINHALA_WORD_BUDGET = 60
    SINHALA_MAX_EXCERPTS = 5
    # Conservative spoken-Sinhala boundary cues, not a linguistic parser.
    SINHALA_FINITE_ENDINGS = (
        "ඉන්නවා", "සිටිනවා", "සිටියි", "තිබෙනවා", "තියෙනවා",
        "වෙනවා", "වෙන්නේ", "කරනවා", "කළා", "වුණා", "වෙනවා",
        "බෑ", "නෑ", "නැහැ", "නොවේ", "නොවෙයි", "නොහැක",
    )
    def __init__(self):
        self.accept_ai_summary = False
        self.accept_english_ai_summary = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Summarizer device:", self.device)
        # Load only the model needed for the selected language, not both.
        self.tokenizer = None
        self.model = None
        self.active_model_name = None
    def load_model(self, name):
        if getattr(self, "active_model_name", None) == name and self.model is not None:
            return
        print("Loading summarization model:", name)
        # Release the previous model before loading a different one.
        self.model = None
        self.tokenizer = None
        self.active_model_name = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        model.to(self.device)
        model.eval()
        self.tokenizer, self.model = tokenizer, model
        self.active_model_name = name
        print("Summarization model loaded successfully.")
    def clean_text(self, text):
        return re.sub(r"\s+", " ", text).strip() if text else ""
    def split_sentences(self, text):
        # Basic punctuation splitting; never cut an idea every 35 words.
        # Abbreviations and unpunctuated ASR need better segmentation.
        text = self.clean_text(text)
        return [s.strip() for s in re.split(r"(?<=[.!?।෴])\s+", text) if s.strip()]
    def get_words(self, text):
        return re.findall(r"[A-Za-z0-9%]+|[\u0D80-\u0DFF]+", text.lower())
    def sentence_similarity(self, sentence1, sentence2):
        words1, words2 = set(self.get_words(sentence1)), set(self.get_words(sentence2))
        return len(words1 & words2) / len(words1 | words2) if words1 and words2 else 0.0
    def sinhala_clauses(self, text):
        """Extract contiguous source spans at cautious predicate boundaries.
        Never split by word count. Keep reporting/conditional continuations
        attached, since e.g. 'නෑ කියලා' cannot safely become 'නෑ'.
        Unrecognized boundaries remain intact rather than being guessed.
        """
        text = self.clean_text(text)
        endings = "|".join(map(re.escape, self.SINHALA_FINITE_ENDINGS))
        predicate = re.compile(r"(?<![\u0D80-\u0DFF])(?:" + endings + r")(?![\u0D80-\u0DFF])")
        continuations = re.compile(
            r"^(?:කියලා|කියා|කියන|බව|බවට|නම්|නිසා|නමුත්|හැබැයි|එහෙත්|වුණත්|වුනත්|නේද|ද)(?:\s|$)"
        )
        # Without punctuation, split only before an explicit new subject cue.
        # This small baseline vocabulary is configurable; it is not universal.
        subject = re.compile(
            r"^(?:මේ\s+(?:රටේ|ආකාරයට)|රජයේ\s+සේවක|ගොවිය|ගොවී|"
            r"පොලිස්\s+සේවාව|සෞඛ්‍ය\s+සේවාව|තරුණ\s+පරපුර|"
            r"ජනතාව|ආණ්ඩුව|රජය|ඔවුන්|අපි|මම)"
        )
        clauses = []
        for sentence in self.split_sentences(text):
            start = 0
            for match in predicate.finditer(sentence):
                tail = sentence[match.end():]
                separator = re.match(r"[\s,;.!?।෴]+", tail)
                if not separator:
                    continue
                next_start = match.end() + separator.end()
                following = sentence[next_start:]
                if not following or continuations.match(following):
                    continue
                marked = any(c in separator.group() for c in ",;.!?।෴")
                if marked or subject.match(following):
                    clauses.append(sentence[start:match.end()].strip())
                    start = next_start
            remainder = sentence[start:].strip()
            if remainder:
                clauses.append(remainder)
        return clauses
    def sinhala_extractive_summary(self, transcript):
        """Return explicitly attributed excerpts, not generated assertions."""
        clauses = self.sinhala_clauses(transcript)
        stopwords = {
            "මේ", "රටේ", "මේක", "ඒ", "සහ", "හා", "එක", "ටික",
            "කරමින්", "ඉන්නවා", "සිටිනවා", "සම්පූර්ණයෙන්", "සම්පූර්ණය",
        }
        endings = "|".join(map(re.escape, self.SINHALA_FINITE_ENDINGS))
        completed = re.compile(r"(?:" + endings + r")[.!?।෴]*$")
        # Exclude dangling list items and vague remarks. Never 'repair' ASR
        # using a guessed sentence, entity, number or political accusation.
        useful = [
            c for c in clauses
            if len(c.split()) >= 4 and completed.search(c)
            and len(set(self.get_words(c)) - stopwords) >= 3
            # Long unpunctuated spans may merge unrelated ASR claims.
            and (re.search(r"[.!?।෴]$", c) or len(c.split()) <= 10)
        ]
        if not useful:
            # No reliable clause ends: retain source rather than invent them.
            excerpt = self.extractive_summary(transcript)
            return 'කථිකයාගේ ප්‍රකාශයෙන් උපුටාගත් කොටස්: “' + excerpt + '”'
        words = [set(self.get_words(c)) - stopwords for c in useful]
        frequencies = Counter(w for row in words for w in row)
        relevance = [sum(frequencies[w] for w in row) / max(1, len(row)) for row in words]
        scale = max(relevance) or 1.0
        selected = []
        # Include opening context when it fits, rather than forcing a huge
        # opening sentence into every summary.
        if len(useful[0].split()) <= self.SINHALA_WORD_BUDGET:
            selected = [0]
        used = sum(len(useful[i].split()) for i in selected)
        while len(selected) < self.SINHALA_MAX_EXCERPTS:
            covered = set().union(*(words[i] for i in selected))
            choices = []
            for i, clause in enumerate(useful):
                if i in selected or used + len(clause.split()) > self.SINHALA_WORD_BUDGET:
                    continue
                if any(self.sentence_similarity(clause, useful[j]) >= 0.65 for j in selected):
                    continue
                novelty = len(words[i] - covered) / max(1, len(words[i]))
                # Select by relevance and new vocabulary; the word budget
                # limits length without favoring tiny excerpts.
                score = 0.35 * relevance[i] / scale + 0.65 * novelty
                choices.append((score, -i, i))
            if not choices:
                break
            i = max(choices)[2]
            selected.append(i)
            used += len(useful[i].split())
        if not selected:
            selected = [0]  # Soft budget: never truncate a complete clause.
        excerpts = [useful[i] for i in sorted(selected)]
        return 'කථිකයාගේ ප්‍රකාශයෙන් උපුටාගත් කොටස්: ' + '; '.join('“' + c + '”' for c in excerpts)
    def extractive_summary(self, text):
        """Select salient, distinct source sentences in their original order.
        Word diversity is only an approximation to topic diversity. The
        word budget never cuts a sentence; one long opening may exceed it.
        """
        sentences = self.split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) == 1:
            # Preserve the statement rather than cutting its qualifications.
            return sentences[0]
        stopwords = {
            "the", "a", "an", "and", "or", "is", "are", "was", "were",
            "to", "of", "in", "on", "for", "that", "this", "it", "with",
            "as", "at", "from", "be", "by", "but", "if", "then",
            "අපි", "ඔයා", "ඒ", "මේ", "එක", "එතකොට", "කියලා",
            "කියන්නේ", "වගේ", "සහ", "හා", "නම්", "තමන්", "මම", "එහෙම", "ඉතින්",
        }
        words = [set(self.get_words(s)) - stopwords for s in sentences]
        frequencies = Counter(w for row in words for w in row)
        def relevance(index):
            row = words[index]
            return sum(frequencies[w] for w in row) / max(1, len(row))
        relevance_scores = [relevance(i) for i in range(len(sentences))]
        scale = max(relevance_scores) or 1.0
        selected = [0]
        used_words = len(sentences[0].split())
        # Keep fewer sentences than the source wherever possible.
        target = min(self.MAX_SUMMARY_SENTENCES, len(sentences) - 1)
        while len(selected) < target:
            covered = set().union(*(words[i] for i in selected))
            candidates = []
            for index in range(1, len(sentences)):
                if index in selected:
                    continue
                if used_words + len(sentences[index].split()) > self.SUMMARY_WORD_BUDGET:
                    continue
                if any(self.sentence_similarity(sentences[index], sentences[i]) >= 0.65
                       for i in selected):
                    continue
                novelty = len(words[index] - covered) / max(1, len(words[index]))
                # Prefer new vocabulary rather than repeating the opening.
                score = 0.35 * relevance_scores[index] / scale + 0.65 * novelty
                candidates.append((score, -index, index))
            if not candidates:
                break
            index = max(candidates)[2]
            selected.append(index)
            used_words += len(sentences[index].split())
        return " ".join(sentences[i] for i in sorted(selected))
    def split_into_chunks(self, text, max_tokens=400):
        """Token chunks for metadata; these are not individually summarized."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [
            chunk for i in range(0, len(ids), max_tokens)
            if (chunk := self.tokenizer.decode(ids[i:i + max_tokens], skip_special_tokens=True).strip())
        ]
    def summarize_chunk(self, text):
        text = self.clean_text(text)
        if not text:
            return ""
        if self.model is None:
            self.load_model(self.MODEL_NAME)
        if len(self.tokenizer.encode(text, add_special_tokens=True)) > self.INPUT_LIMIT:
            raise ValueError("Evidence exceeds the mT5 input limit; no text was truncated.")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                # Override CNN checkpoint minimums; do not force padding a
                # short transcript into an unnecessarily long summary.
                max_length=84,
                max_new_tokens=None,
                min_length=0,
                min_new_tokens=None,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        return self.clean_text(self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ))
    def groundedness_score(self, transcript, summary):
        """Lexical precision, not a probability of factual correctness."""
        source = set(self.get_words(transcript))
        words = self.get_words(summary)
        return sum(w in source for w in words) / len(words) if words else 0.0
    def coverage_score(self, evidence, summary):
        """Lexical recall, not coverage of important ideas."""
        source, words = set(self.get_words(evidence)), set(self.get_words(summary))
        return len(source & words) / len(source) if source else 0.0
    def numbers_are_grounded(self, transcript, summary):
        # Digit matching does not validate written quantities, units or roles.
        pattern = r"\d+(?:\\.\d+)?%?"
        return set(re.findall(pattern, summary)).issubset(re.findall(pattern, transcript))
    def negation_is_consistent(self, source, candidate):
        """Compare global negation presence; does not establish entailment."""
        if not source or not candidate:
            return False
        english = {
            "not", "no", "never", "cannot", "can't", "cant", "don't", "dont",
            "doesn't", "doesnt", "didn't", "didnt", "shouldn't", "shouldnt",
            "won't", "wont", "without", "isn't", "isnt", "aren't", "arent",
            "wasn't", "wasnt", "weren't", "werent", "couldn't", "couldnt",
        }
        sinhala = {"බෑ", "නෑ", "නැහැ", "නෙමෙයි", "නොවේ", "නොවෙයි", "නොහැක", "එපා"}
        def has_negation(text):
            text = text.casefold().replace("’", "'")
            en_tokens = set(re.findall(r"[a-z]+(?:'[a-z]+)?", text))
            si_tokens = re.findall(r"[\u0D80-\u0DFF]+", text)
            return bool(en_tokens & english) or any(
                word in sinhala or word.startswith("නො") for word in si_tokens
            )
        return has_negation(source) == has_negation(candidate)
    def summarize_evidence_sentences(self, evidence):
        # Compatibility helper: generate once from the complete evidence.
        # Never replace rejected AI sentences inside the raw candidate.
        return self.summarize_chunk(evidence)
    def summarize(self, transcript, language=None):
        transcript = self.clean_text(transcript)
        result = {
            "language": language,
            "summary": "",
            "summary_evidence": "",
            "ai_candidate_summary": "",
            "summary_method": "none",
            "groundedness_score": 0.0,
            "coverage_score": 0.0,
            "negation_consistent": False,
            "chunk_count": 0,
            "model": self.MODEL_NAME,
        }
        if not transcript:
            return result
        is_sinhala = language == "si" or (
            language is None and bool(re.search(r"[\u0D80-\u0DFF]", transcript))
        )
        is_english = language == "en" or (
            language is None and not is_sinhala and bool(re.search(r"[A-Za-z]", transcript))
        )
        fallback = self.sinhala_extractive_summary(transcript) if is_sinhala else self.extractive_summary(transcript)
        selected_model = self.ENGLISH_MODEL_NAME if is_english else self.MODEL_NAME
        try:
            self.load_model(selected_model)
        except (OSError, RuntimeError) as exc:
            print("Summarization model unavailable; using source extraction:", exc)
            result.update({
                "summary": fallback,
                "summary_evidence": transcript,
                "summary_method": "extractive_fallback",
                # This names the attempted model; chunk_count remains zero
                # because no tokenizer could be loaded to compute it.
                "model": selected_model,
            })
            return result
        chunks = self.split_into_chunks(transcript)
        # Use all context whenever it fits. Long transcripts use the
        # selected sentences; this is not exhaustive long-video coverage.
        full_length = len(self.tokenizer.encode(transcript, add_special_tokens=True))
        # Attribution labels/quote marks must not become model evidence.
        source_evidence = " ".join(self.sinhala_clauses(transcript)) if is_sinhala else fallback
        evidence = transcript if full_length <= self.INPUT_LIMIT else source_evidence
        evidence_length = len(self.tokenizer.encode(evidence, add_special_tokens=True))
        print("\nSelected summary evidence:")
        print(evidence)
        candidate = ""
        if evidence_length <= self.INPUT_LIMIT:
            try:
                candidate = self.clean_text(self.summarize_chunk(evidence))
            except RuntimeError as exc:
                print("mT5 generation failed; using extraction:", exc)
        else:
            print("Selected evidence exceeds the input limit; using extraction.")
        grounded = self.groundedness_score(evidence, candidate)
        coverage = self.coverage_score(evidence, candidate)
        numbers_ok = self.numbers_are_grounded(evidence, candidate)
        negation_ok = self.negation_is_consistent(evidence, candidate)
        shorter = bool(candidate) and len(candidate) < len(evidence) and (
            len(candidate.split()) < len(evidence.split())
        )
        if is_english:
            # Coverage is reported rather than used as a minimum: concise
            # summaries legitimately omit source vocabulary. Conservative
            # precision and output-length checks still do not verify meaning.
            source_names = set(re.findall(r"\b[A-Z][A-Za-z]+\b", evidence))
            candidate_sentences = self.split_sentences(candidate)
            candidate_names = set()
            for sentence in candidate_sentences:
                # Ignore a capitalized first word caused by sentence position.
                rest = sentence.split(maxsplit=1)
                if len(rest) == 2:
                    candidate_names.update(re.findall(r"\b[A-Z][A-Za-z]+\b", rest[1]))
            names_ok = candidate_names.issubset(source_names)
            screened = (
                shorter and grounded >= 0.85 and numbers_ok and negation_ok
                and names_ok and 1 <= len(candidate_sentences) <= 3
                and len(candidate.split()) <= 60
            )
            enabled = getattr(self, "accept_english_ai_summary", False)
        else:
            screened = (
                shorter and grounded >= 0.70 and coverage >= 0.35
                and numbers_ok and negation_ok
            )
            enabled = getattr(self, "accept_ai_summary", False)
        use_ai = enabled and screened
        print("\nRaw AI candidate:", candidate)
        print("Candidate lexical groundedness:", round(grounded, 3))
        print("Candidate lexical coverage:", round(coverage, 3))
        print("Candidate digit check:", numbers_ok)
        print("Candidate global negation check:", negation_ok)
        print("Candidate passes lexical screening:", screened)
        print("AI acceptance enabled:", enabled)
        result.update({
            "summary": candidate if use_ai else fallback,
            "summary_evidence": evidence,
            "ai_candidate_summary": candidate,
            "summary_method": (
                "english_ai_heuristic_summary" if use_ai and is_english
                else "mt5_lexically_screened" if use_ai
                else "extractive_fallback"
            ),
            # These diagnostics always describe the raw candidate, even
            # when the returned summary is extractive.
            "groundedness_score": grounded,
            "coverage_score": coverage,
            "negation_consistent": negation_ok,
            "chunk_count": len(chunks),
            "model": selected_model,
        })
        print("Summary method:", result["summary_method"])
        return result
