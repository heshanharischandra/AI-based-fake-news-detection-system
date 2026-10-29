from faster_whisper import WhisperModel


class SpeechToText:
    def __init__(self):
        print("Loading Whisper model...")
        self.model = WhisperModel(
            "large-v3",
            device="cpu",
            compute_type="int8",
        )
        print("Whisper model loaded.")

    def transcribe(self, audio_path, language=None):
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0,
        )
        transcript = []
        segment_data = []
        for segment in segments:
            text = segment.text.strip()
            transcript.append(text)
            segment_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
            })
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": " ".join(transcript),
            "segments": segment_data,
        }
