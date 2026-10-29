"""Lightweight integration tests; no model downloads or video processing.

Run from project root: python test_upload_pipeline.py
"""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from app.services.upload_pipeline import process_video


def module(name, **attributes):
    value = types.ModuleType(name)
    value.__dict__.update(attributes)
    return value


class UploadPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.video = self.base / 'arbitrary_upload.mp4'
        self.video.write_bytes(b'test input; media decoding is mocked')
        self.text = {
            'en': 'The hospital reported that 12 patients were not admitted on Saturday.',
            'si': 'ගොවියොන්ට පෝර ටික නෑ',
        }
        owner = self

        class FakeEnglishSpeech:
            def transcribe(self, path, language):
                assert Path(path).is_file()
                assert language == 'en'
                return {'language': 'en', 'text': owner.text['en'], 'segments': []}

        class FakeSinhalaSpeech:
            def transcribe(self, path, return_segments):
                assert Path(path).is_file()
                assert return_segments is True
                return {'language': 'si', 'text': owner.text['si'], 'segments': []}

            @staticmethod
            def add_claim_timestamps(claims, result):
                for claim in claims:
                    claim.update(start=1.0, end=2.0, timestamp_status='test_fixture')

        class FakeSummary:
            def summarize(self, transcript, language):
                return {'summary': transcript, 'language': language, 'summary_method': 'test_fixture'}

        class FakeNER:
            def enrich_claims(self, claims, language):
                for claim in claims:
                    claim['entities'] = {'language': language, 'locations': []}
                return claims

        def audio(video, output):
            assert Path(video) == owner.video.resolve()
            Path(output).write_bytes(b'audio fixture')

        def frames(video, folder, interval_seconds):
            assert Path(video) == owner.video.resolve()
            assert interval_seconds == 5
            Path(folder).mkdir()
            return 2

        replacements = {
            'app.services.video_processor': module('video_processor', extract_audio=audio, extract_frames=frames),
            'app.services.speech_to_text': module('speech_to_text', SpeechToText=FakeEnglishSpeech),
            'app.services.sinhala_speech_to_text': module('sinhala_speech_to_text', SinhalaSpeechToText=FakeSinhalaSpeech),
            'app.services.summarizer': module('summarizer', VideoSummarizer=FakeSummary),
            'app.services.ner_extractor': module('ner_extractor', NERExtractor=FakeNER),
        }
        self.patch = patch.dict('sys.modules', replacements)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_both_languages_preserve_review_entities_queries_and_json(self):
        for language in ('si', 'en'):
            with self.subTest(language=language):
                output = self.base / language
                stages = []
                result = process_video(self.video, language, output,
                                       on_progress=lambda number, label: stages.append(number),
                                       original_name='original.mp4')
                self.assertEqual(stages, list(range(1, 9)))
                self.assertEqual(json.loads((output / 'result.json').read_text(encoding='utf-8')), result)
                self.assertEqual((output / 'transcript.txt').read_text(encoding='utf-8'), self.text[language])
                self.assertEqual(result['upload_details']['original_filename'], 'original.mp4')
                self.assertEqual(result['upload_details']['frame_count'], 2)
                claims = result['claim_analysis']['claims']
                self.assertEqual(len(claims), 1)
                claim = claims[0]
                self.assertIn('review_reasons', claim)
                self.assertIn('entities', claim)
                self.assertEqual(claim['search_queries']['source_text'], claim['text'])
                if language == 'si':
                    self.assertEqual(claim['start'], 1.0)
                    self.assertIn('නෑ', claim['search_queries']['queries'][0]['query'])
                else:
                    self.assertEqual(claim['search_queries']['status'], 'draft_unscreened')

    def test_existing_output_is_not_overwritten(self):
        folder = self.base / 'existing'
        folder.mkdir()
        sentinel = folder / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            process_video(self.video, 'en', folder)
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_empty_claim_list_skips_ner(self):
        self.text['en'] = ''
        result = process_video(self.video, 'en', self.base / 'empty')
        self.assertEqual(result['claim_analysis']['claims'], [])

    def test_validation_and_failures_do_not_save_success_results(self):
        with self.assertRaises(ValueError):
            process_video(self.video, 'xx', self.base / 'bad_language')
        with self.assertRaises(ValueError):
            process_video(self.base / 'missing.mp4', 'en', self.base / 'missing')
        output = self.base / 'failure'
        with patch('app.services.video_processor.extract_audio', side_effect=RuntimeError('fixture failure')):
            with self.assertRaises(RuntimeError):
                process_video(self.video, 'en', output)
        self.assertFalse((output / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
