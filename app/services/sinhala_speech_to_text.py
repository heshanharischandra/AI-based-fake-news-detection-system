"""Sinhala ASR plus forced alignment of its unchanged transcript.

Install alignment support: python -m pip install "stable-ts[fw]"
"""
import gc
import math
import re
import wave

from transformers import pipeline

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def validate_alignment(data, transcript, duration):
    """Reject changed/missing text and invalid times before saving anything."""
    segments = data.get("segments", [])
    if not segments:
        raise ValueError("The aligner returned no segments.")
    expected = clean(transcript)
    joined = clean("".join(s.get("text", "") for s in segments))
    if joined != expected:
        # Some versions store stripped segment text. Allow spaces between
        # segments, but never allow missing or rewritten transcript words.
        joined = clean(" ".join(s.get("text", "") for s in segments))
    if joined != expected:
        raise ValueError("Aligned text differs from the saved transcript; output was not saved.")

    result = []
    zero_duration_words = 0
    previous_end = 0.0
    for segment in segments:
        start, end = segment.get("start"), segment.get("end")
        if not (finite(start) and finite(end) and 0 <= start < end <= duration + 0.05):
            raise ValueError("An aligned segment has invalid/out-of-range timestamps.")
        if start < previous_end - 0.05:
            raise ValueError("Aligned segments overlap or are out of order.")
        words = segment.get("words", [])
        if not words:
            raise ValueError("An aligned segment has no word timestamps.")
        last_word_end = start
        word_text = clean("".join(w.get("word", "") for w in words))
        if word_text != clean(segment.get("text", "")):
            word_text = clean(" ".join(w.get("word", "").strip() for w in words))
        if word_text != clean(segment.get("text", "")):
            raise ValueError("Word timestamps do not cover the segment text.")
        normalized_words = []
        for word in words:
            ws, we = word.get("start"), word.get("end")
            if not (finite(ws) and finite(we) and start - 0.05 <= ws <= we <= end + 0.05):
                raise ValueError("An aligned word has invalid timestamps.")
            if ws < last_word_end - 0.05:
                raise ValueError("Aligned words overlap or are out of order.")
            if we == ws:
                zero_duration_words += 1
            normalized_words.append({
                "word": word["word"], "start": round(ws, 3), "end": round(we, 3),
                "timestamp_status": "unresolved" if we == ws else "estimated",
            })
            last_word_end = we
        result.append({
            "start": round(start, 3), "end": round(end, 3),
            "text": segment["text"].strip(), "words": normalized_words,
        })
        previous_end = end
    return result, zero_duration_words

def add_claim_times(claims, segments):
    """Attach enclosing word times only for unique verbatim matches.

    Rewritten or ambiguous claims remain explicitly unmapped. A long claim
    receives a long time interval; this does not split it into atomic claims.
    """
    words = [w for segment in segments for w in segment["words"]]
    text = ""
    spans = []
    for word in words:
        part = clean(word["word"])
        if not part:
            continue
        if text:
            text += " "
        start = len(text)
        text += part
        spans.append((start, len(text), word))

    for claim in claims:
        needle = clean(claim.get("text", ""))
        matches = list(re.finditer(re.escape(needle), text)) if needle else []
        claim.update({"start": None, "end": None, "timestamp_status": "unmapped"})
        if len(matches) != 1:
            continue
        match = matches[0]
        covering = [w for a, b, w in spans if a < match.end() and b > match.start()]
        if covering:
            if covering[-1]["end"] <= covering[0]["start"]:
                continue
            claim.update({
                "start": covering[0]["start"], "end": covering[-1]["end"],
                "timestamp_status": (
                    "estimated_with_unresolved_words"
                    if any(w["start"] == w["end"] for w in covering)
                    else "estimated_from_alignment"
                ),
            })


class SinhalaSpeechToText:
    MODEL_NAME = "SinhaSpeech/whisper-small-sinhala-v6-e6-run11-best"
    ALIGNMENT_MODEL = "large-v3"

    def __init__(self):
        print("Loading Sinhala ASR model...")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.MODEL_NAME,
            device=-1,
            chunk_length_s=30,
        )
        print("Sinhala ASR model loaded.")

    def _align(self, audio_path, transcript):
        try:
            import stable_whisper
        except ImportError as exc:
            raise RuntimeError(
                'Install the alignment dependency: python -m pip install "stable-ts[fw]"'
            ) from exc
        # The existing pipeline creates PCM WAV audio. Use its actual length
        # to validate model-estimated timestamps rather than inventing times.
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        print("Loading Sinhala alignment model:", self.ALIGNMENT_MODEL)
        aligner = stable_whisper.load_faster_whisper(
            self.ALIGNMENT_MODEL, device="cpu", compute_type="int8",
        )
        try:
            lines = re.split(r"(?<=[.!?।෴])\s+", clean(transcript))
            print("Aligning the existing transcript to the audio...")
            aligned = aligner.align(
                str(audio_path), "\n".join(lines), language="si",
                original_split=True, remove_instant_words=False, verbose=False,
                # Tested on the current Sinhala clip: a larger token window
                # avoids the default alignment's duplicate/missing words.
                # Silence adjustment was compressing late text to the end.
                token_step=300, suppress_silence=False,
            )
            if aligned is None:
                raise ValueError("The aligner returned no result.")
            return validate_alignment(aligned.to_dict(), transcript, duration)
        finally:
            # Release this extra large model before loading the summarizer.
            del aligner
            gc.collect()

    def transcribe(self, audio_path, return_segments=True):
        """Transcribe once, then optionally align that exact text to audio.

        return_segments=False retains text-only operation. Alignment does not
        run the fine-tuned model's previously failing timestamp processor.
        Estimated word/segment times still require listening-based review.
        """
        print("Starting Sinhala transcription...")
        asr_result = self.pipe(
            str(audio_path),
            generate_kwargs={"language": "sinhala", "task": "transcribe"},
        )
        transcript = asr_result.get("text", "").strip()
        result = {
            "language": "si",
            "text": transcript,
            "segments": ([{"start": None, "end": None, "text": transcript}]
                         if transcript else []),
            "timestamp_mode": "disabled",
            "timestamp_details": {"status": "disabled", "estimated": False},
        }
        print("Sinhala transcription finished.")
        if transcript and return_segments:
            try:
                segments, zero_words = self._align(audio_path, transcript)
            except Exception as exc:
                # Alignment is optional enrichment. Never lose the working
                # transcription or fabricate timestamps when it fails.
                result["timestamp_mode"] = "unavailable"
                result["timestamp_details"] = {
                    "status": "failed", "model": self.ALIGNMENT_MODEL,
                    "estimated": False, "error": str(exc),
                }
                print("Sinhala alignment failed:", exc)
                print("Continuing with the original transcript; timestamps are unavailable.")
            else:
                result["segments"] = segments
                result["timestamp_mode"] = "forced_alignment"
                result["timestamp_details"] = {
                    "status": "complete", "model": self.ALIGNMENT_MODEL,
                    "library": "stable-ts", "estimated": True,
                    "zero_duration_word_count": zero_words,
                    "review_required": True,
                    "note": "Review against audio, especially where ASR text is incorrect.",
                }
        print("Timestamp mode:", result["timestamp_mode"])
        print("Segment count:", len(result["segments"]))
        return result

    @staticmethod
    def add_claim_timestamps(claims, transcription_result):
        """Add times after claim analysis; do not alter claim text or scores."""
        if transcription_result.get("timestamp_mode") == "forced_alignment":
            add_claim_times(claims, transcription_result["segments"])
        else:
            for claim in claims:
                claim.update({"start": None, "end": None, "timestamp_status": "unavailable"})
