from app.services.video_processor import extract_audio, extract_frames
from app.services.speech_to_text import SpeechToText
from app.services.sinhala_speech_to_text import SinhalaSpeechToText
from app.services.summarizer import VideoSummarizer
from app.services.claim_analyzer import ClaimAnalyzer
from app.services.ner_extractor import NERExtractor
from app.services.query_generator import QueryGenerator

from pathlib import Path
import json
import sys


# =========================================================
# SELECT LANGUAGE
# =========================================================

# Run:
# python test_pipeline.py en
# python test_pipeline.py si

LANGUAGE = sys.argv[1].lower() if len(sys.argv) > 1 else "en"


# =========================================================
# INPUT / OUTPUT PATHS
# =========================================================

if LANGUAGE == "en":

    VIDEO_PATH = "data/videos/eng_test.mp4"
    AUDIO_PATH = "data/audio/eng_test.wav"
    FRAME_FOLDER = "data/frames/eng_test"

    TRANSCRIPT_PATH = Path(
        "data/transcripts/eng_test.txt"
    )

    JSON_PATH = "data/transcripts/eng_test.json"


elif LANGUAGE == "si":

    VIDEO_PATH = "data/videos/sinhala_updated.mp4"
    AUDIO_PATH = "data/audio/sinhala_updated.wav"
    FRAME_FOLDER = "data/frames/sinhala_updated"

    TRANSCRIPT_PATH = Path(
        "data/transcripts/sinhala_updated.txt"
    )

    JSON_PATH = "data/transcripts/sinhala_updated.json"


else:

    print("Invalid language.")
    print("Use:")
    print("python test_pipeline.py en")
    print("or")
    print("python test_pipeline.py si")

    sys.exit()


print("\n===================================")
print("Starting video processing...")
print("Language:", LANGUAGE)
print("===================================\n")


# =========================================================
# STEP 1 - EXTRACT AUDIO
# =========================================================

print("STEP 1: Extracting audio")

extract_audio(
    VIDEO_PATH,
    AUDIO_PATH
)

print("Audio extraction completed.")


# =========================================================
# STEP 2 - EXTRACT FRAMES
# =========================================================

print("\nSTEP 2: Extracting frames")

frame_count = extract_frames(
    VIDEO_PATH,
    FRAME_FOLDER,
    interval_seconds=5
)

print("Frames:", frame_count)


# =========================================================
# STEP 3 - SPEECH TO TEXT
# =========================================================

print("\nSTEP 3: Speech-to-text")


# ---------------------------------------------------------
# ENGLISH
# ---------------------------------------------------------

if LANGUAGE == "en":

    print(
        "Using English / Multilingual Whisper model..."
    )

    speech_model = SpeechToText()

    result = speech_model.transcribe(
        AUDIO_PATH,
        language="en"
    )


# ---------------------------------------------------------
# SINHALA
# ---------------------------------------------------------

elif LANGUAGE == "si":

    print(
        "Using Sinhala-specific ASR model..."
    )

    speech_model = SinhalaSpeechToText()

    result = speech_model.transcribe(
        AUDIO_PATH,
        return_segments=True
    )


# =========================================================
# DISPLAY TRANSCRIPTION RESULT
# =========================================================

print("\nDetected Language:")
print(result["language"])


# English model provides language probability.
# Sinhala-specific model may not provide it.

if "language_probability" in result:

    print("\nLanguage Confidence:")
    print(result["language_probability"])


print("\nTranscript:")
print(result["text"])


# =========================================================
# STEP 4 - VIDEO SUMMARIZATION
# =========================================================

print("\nSTEP 4: Video summarization")

# Inspect ASR boundaries before using them to select summary text.
print("\nASR timestamp mode:", result.get("timestamp_mode", "provided_by_whisper"))
print("ASR segment count:", len(result.get("segments", [])))
for segment in result.get("segments", []):
    print(segment.get("start"), "->", segment.get("end"), ":", segment.get("text", ""))

summarizer = VideoSummarizer()

summary_result = summarizer.summarize(
    transcript=result["text"],
    language=result["language"]
)


print("\nVideo Summary:")
print(
    summary_result["summary"]
)


# Add summary information into final result.

result["summary"] = (
    summary_result["summary"]
)

result["summary_details"] = (
    summary_result
)


# =========================================================
# STEP 5 - CLAIM EXTRACTION
# =========================================================

print("\nSTEP 5: Claim extraction")

claim_analyzer = ClaimAnalyzer()

claim_result = claim_analyzer.analyze(
    transcript=result["text"],
    language=result["language"]
)


print(
    "\nCheck-worthy claims found:",
    claim_result["claim_count"]
)


for claim in claim_result["claims"]:

    print(
        f"\nClaim {claim['claim_id']}:"
    )

    print(
        claim["text"]
    )

    print(
        "Score:",
        claim["check_worthy_score"]
    )

    print(
        "Reasons:",
        ", ".join(
            claim["reasons"]
        )
    )


# =========================================================
# STEP 6 - NAMED ENTITY RECOGNITION
# =========================================================

print(
    "\nSTEP 6: Named Entity Recognition"
)


# Only load the large NER model if
# check-worthy claims were found.

if claim_result["claim_count"] > 0:

    ner_extractor = NERExtractor()

    enriched_claims = (
        ner_extractor.enrich_claims(
            claims=claim_result["claims"],
            language=result["language"]
        )
    )


    # Replace the original claims with
    # the NER-enriched claims.

    claim_result["claims"] = (
        enriched_claims
    )


    print(
        "\nNER enriched claims:",
        len(enriched_claims)
    )


    for claim in enriched_claims:

        print(
            f"\nClaim {claim['claim_id']}:"
        )

        print(
            claim["text"]
        )

        entities = claim["entities"]


        print(
            "Persons:",
            entities["persons"]
        )

        print(
            "Organizations:",
            entities["organizations"]
        )

        print(
            "Locations:",
            entities["locations"]
        )

        print(
            "Dates:",
            entities["dates"]
        )

        print(
            "Numbers:",
            entities["numbers"]
        )

        print(
            "Miscellaneous:",
            entities["miscellaneous"]
        )


else:

    print(
        "No check-worthy claims found. "
        "Skipping NER."
    )


# =========================================================
# ADD CLAIM ANALYSIS TO FINAL RESULT
# =========================================================

# This now contains:
#
# claim_id
# claim text
# check-worthiness score
# reasons
# NER entities
#
# if NER was performed.

if LANGUAGE == "si":
    speech_model.add_claim_timestamps(claim_result["claims"], result)

# STEP 7: Generate local search drafts; no web requests are made.
print("\nSTEP 7: Search query generation")
query_generator = QueryGenerator()
claim_result["claims"] = query_generator.enrich_claims(
    claim_result["claims"], language=result["language"]
)
for claim in claim_result["claims"]:
    drafts = claim["search_queries"]
    print(f"\nClaim {claim['claim_id']}: {drafts['status']}")
    for query in drafts["queries"]:
        print(query["kind"] + ":", query["query"])

result["claim_analysis"] = (
    claim_result
)


# =========================================================
# SAVE TXT TRANSCRIPT
# =========================================================

TRANSCRIPT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


TRANSCRIPT_PATH.write_text(
    result["text"],
    encoding="utf-8"
)


print(
    "\nTranscript saved:",
    TRANSCRIPT_PATH
)


# =========================================================
# SAVE JSON
# =========================================================

with open(
    JSON_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        ensure_ascii=False,
        indent=4
    )


print(
    "JSON saved:",
    JSON_PATH
)


# =========================================================
# FINISHED
# =========================================================

print("\n===================================")
print("Video processing completed.")
print("===================================")
