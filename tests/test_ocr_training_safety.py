import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_sinhala_ocr import audit_splits, can_reuse_encoding, promotion_check, training_inputs_match, validate_resume_splits
from scripts.evaluate_ocr_runtime import evaluate_records, use_tessdata
from app.services.ocr_service import extract_text


class OCRTrainingSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def record(self, name, text, content):
        image = self.root / name
        image.write_bytes(content)
        return {"image": str(image), "text": text}

    def test_disjoint_splits_pass(self):
        rows = {split: [self.record(split + '.png', split, split.encode())]
                for split in ('train', 'validation', 'test')}
        self.assertEqual(audit_splits(rows)['status'], 'passed')

    def test_duplicate_labels_or_image_content_fail(self):
        for duplicate_image in (True, False):
            rows = {split: [self.record(split + '.png', split, split.encode())]
                    for split in ('train', 'validation', 'test')}
            if duplicate_image:
                Path(rows['test'][0]['image']).write_bytes(b'train')
            else:
                rows['test'][0]['text'] = ' TRAIN '
            with self.assertRaises(ValueError):
                audit_splits(rows)

    def test_empty_split_fails(self):
        with self.assertRaises(ValueError):
            audit_splits({'train': [], 'validation': [], 'test': []})

    def test_encoding_requires_matching_labels_and_existing_inputs(self):
        image = self.root / 'line.png'
        image.write_bytes(b'image')
        self.assertFalse(can_reuse_encoding(image, 'headline'))
        image.with_suffix('.gt.txt').write_text('headline\n', encoding='utf-8')
        image.with_suffix('.box').write_text('WordStr 0 0 50 10 0 #headline\n', encoding='utf-8')
        image.with_suffix('.lstmf').write_bytes(b'encoded')
        self.assertTrue(can_reuse_encoding(image, 'headline'))
        self.assertFalse(can_reuse_encoding(image, 'changed headline'))
        image.with_suffix('.box').write_text('#different', encoding='utf-8')
        self.assertFalse(can_reuse_encoding(image, 'headline'))

    def test_newer_source_invalidates_cached_encoding(self):
        image = self.root / 'line.png'
        image.write_bytes(b'image')
        image.with_suffix('.gt.txt').write_text('headline', encoding='utf-8')
        image.with_suffix('.box').write_text('WordStr 0 0 50 10 0 #headline', encoding='utf-8')
        image.with_suffix('.lstmf').write_bytes(b'encoded')
        import os
        os.utime(image, ns=(image.stat().st_atime_ns, image.with_suffix('.lstmf').stat().st_mtime_ns + 1_000_000_000))
        self.assertFalse(can_reuse_encoding(image, 'headline'))

    def test_runtime_benchmark_counts_failures_instead_of_dropping_them(self):
        rows = [{'image': 'first.png', 'text': 'correct headline'},
                {'image': 'second.png', 'text': 'another headline'}]
        with patch('scripts.evaluate_ocr_runtime.ocr_service.extract_text', side_effect=[
            {'extracted_text': 'correct headline', 'status': 'success'}, OSError('bad image')
        ]):
            summary, predictions = evaluate_records(rows, workers=1)
        self.assertEqual(summary['samples'], 2)
        self.assertEqual(summary['exact_line_accuracy'], 0.5)
        self.assertEqual(summary['status_counts']['error'], 1)
        self.assertEqual(predictions[1]['prediction'], '')

    def test_model_override_restores_environment_on_error(self):
        import os
        for name in ('sin.traineddata', 'eng.traineddata'):
            (self.root / name).write_bytes(b'fixture')
        with patch.dict(os.environ, {'SINHALA_TESSDATA_DIR': 'previous'}):
            with self.assertRaises(RuntimeError):
                with use_tessdata(self.root):
                    self.assertEqual(os.environ['SINHALA_TESSDATA_DIR'], str(self.root))
                    raise RuntimeError('expected')
            self.assertEqual(os.environ['SINHALA_TESSDATA_DIR'], 'previous')

    def test_promotion_requires_gain_and_no_word_or_line_regression(self):
        current = dict(character_error_rate=0.20, word_error_rate=0.4, exact_line_accuracy=0.1)
        better = dict(character_error_rate=0.15, word_error_rate=0.3, exact_line_accuracy=0.2)
        self.assertTrue(promotion_check(current, better, current, better)[0])
        self.assertFalse(promotion_check(current, current, current, current)[0])
        for change in (dict(character_error_rate=0.3), dict(word_error_rate=0.5),
                       dict(exact_line_accuracy=0.05)):
            self.assertFalse(promotion_check(current, better, current, dict(better, **change))[0])
        self.assertFalse(promotion_check(current, better, current, better, real_regression=0.1)[0])

    def test_runtime_evaluation_uses_requested_segmentation_modes(self):
        with patch('scripts.evaluate_ocr_runtime.ocr_service.extract_text', return_value={
            'status': 'success', 'extracted_text': 'headline'
        }) as extract:
            evaluate_records([{'image': 'fixture.png', 'text': 'headline'}], workers=1, psm_modes=(13,))
        extract.assert_called_once_with('fixture.png', psm_modes=(13,))

    def test_invalid_runtime_modes_fail_before_opening_image(self):
        for modes in ((), (14,), (-1,), (True,), ('13',)):
            with self.assertRaises(ValueError):
                extract_text('not-an-image.png', psm_modes=modes)

    def test_box_label_prefix_is_not_a_matching_label(self):
        image = self.root / 'line.png'
        image.write_bytes(b'image')
        image.with_suffix('.gt.txt').write_text('headline', encoding='utf-8')
        image.with_suffix('.box').write_text('WordStr 0 0 50 10 0 #headline extra', encoding='utf-8')
        self.assertFalse(training_inputs_match(image, 'headline'))

    def test_resume_requires_original_split_lists(self):
        checkpoint = self.root / 'model_checkpoint'
        records = {split: [{'image': str(self.root / (split + '.png'))}]
                   for split in ('train', 'validation')}
        with self.assertRaises(ValueError):
            validate_resume_splits(checkpoint, records)
        for split in records:
            (self.root / (split + '.txt')).write_text(
                str(self.root / (split + '.lstmf')) + '\n', encoding='utf-8')
        validate_resume_splits(checkpoint, records)
        records['train'][0]['image'] = str(self.root / 'test.png')
        with self.assertRaises(ValueError):
            validate_resume_splits(checkpoint, records)


if __name__ == '__main__':
    unittest.main()
