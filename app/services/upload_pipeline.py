"""Reusable local pipeline for an uploaded video; no Streamlit dependency."""

import gc
import json
from pathlib import Path


def process_video(video_path, language, output_dir, on_progress=None, original_name=None):
    """Run existing services and save each analysis in a new output folder.

    The optional callback receives (step_number, step_label), steps 1..8.
    Exceptions propagate to the caller; a failed run is not marked complete.
    """
    if language not in {"si", "en"}:
        raise ValueError("Choose Sinhala or English.")
    video_path = Path(video_path).resolve()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise ValueError("The uploaded video is missing or empty.")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    def report(step, label):
        if on_progress is not None:
            on_progress(step, label)

    from app.services.video_processor import extract_audio, extract_frames

    report(1, "Extracting audio")
    audio_path = output_dir / "audio.wav"
    extract_audio(str(video_path), str(audio_path))

    report(2, "Extracting frames")
    frame_count = extract_frames(str(video_path), str(output_dir / "frames"), interval_seconds=5)

    report(3, "Transcribing speech")
    if language == "si":
        from app.services.sinhala_speech_to_text import SinhalaSpeechToText
        speech = SinhalaSpeechToText()
        result = speech.transcribe(str(audio_path), return_segments=True)
    else:
        from app.services.speech_to_text import SpeechToText
        speech = SpeechToText()
        result = speech.transcribe(str(audio_path), language="en")
    del speech
    gc.collect()

    report(4, "Summarizing the transcript")
    from app.services.summarizer import VideoSummarizer
    summarizer = VideoSummarizer()
    summary = summarizer.summarize(transcript=result["text"], language=result["language"])
    result["summary"] = summary["summary"]
    result["summary_details"] = summary
    del summarizer
    gc.collect()

    report(5, "Extracting candidate claims")
    from app.services.claim_analyzer import ClaimAnalyzer
    analysis = ClaimAnalyzer().analyze(transcript=result["text"], language=result["language"])

    report(6, "Extracting named entities")
    if analysis["claims"]:
        from app.services.ner_extractor import NERExtractor
        ner = NERExtractor()
        analysis["claims"] = ner.enrich_claims(claims=analysis["claims"], language=result["language"])
        del ner
        gc.collect()
    if language == "si":
        SinhalaSpeechToText.add_claim_timestamps(analysis["claims"], result)

    report(7, "Generating search drafts")
    from app.services.query_generator import QueryGenerator
    analysis["claims"] = QueryGenerator().enrich_claims(analysis["claims"], language=result["language"])
    result["claim_analysis"] = analysis
    result["upload_details"] = {
        "original_filename": original_name or video_path.name,
        "selected_language": language,
        "frame_count": frame_count,
        "output_directory": str(output_dir),
    }

    report(8, "Saving results")
    (output_dir / "transcript.txt").write_text(result["text"], encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    return result
