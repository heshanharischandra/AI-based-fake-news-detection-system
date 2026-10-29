"""Run beside summarizer.py: python test_summarizer_regression.py.

Tests selection and return contract without downloading or running models.
"""
import re
from app.services.summarizer import VideoSummarizer

SI = 'මේ වෙලාවේ ආණ්ඩුව ජනතාවගේ ඇස් වහලා, විවිධ කාරණා උඩ ඇස් වහලා මේ රටේ සංස්කෘතිය විනාස කරමින් ඉන්නවා. මේ රටේ මහා සංඝ රත්නයට පහර දෙමින් සිටිනවා. මේ රටේ අධ්‍යාපනය විනාස කරළත් තරුණ පරපුරේ අනාගතය නැති කරමින් ඉන්නවා. සෞඛ්‍ය සේවාව සම්පූර්ණයෙන් විනාස කරලා, රෝහල් වල බේත්වියොන්ට වී ටික විකුණගන්න බෑ ගොවියොන්ට පෝර ටික නෑ රජයේ සේවකයෝ අමාරුවෙන් ජීවත් වෙන්නේ පොලිස් සේවාව සම්පූර්ණය විනාස කරලා දාලා මේ ආකාරයට මේ රටේ සම්පූර්ණ මේ සමාජ පද්ධති කඩා වට්ටමින් විනාස කරමින් ඉන්නවා උන්ගේ දේශපාලන අවශ්‍යතාවයන් මත'
EN = "Colombo has been lit up with lights and color on Saturday as Buddhists across Sri Lanka celebrated Vesak, the island's most important religious festival. Streets were filled with crowds admiring illuminated pandals and giant lanterns depicting seeds from the life of the Buddha. Vesak is observed on the May full moon to commemorate the birth, enlightenment and passing away of the Buddha. Sri Lanka devotees visited temples, meditated, listened to sermons, and lit oil lamps in acts of worship and reflection, although people say the crowd has been lesser this year."
EN_CANDIDATE = 'Vesak is observed on the May full moon to commemorate the birth, enlightenment and passing away of the Buddha . Sri Lanka devotees visited temples, meditated, listened to sermons and lit oil lamps in acts of worship .'
SI_CANDIDATE = 'මේ රටේ සංස්කෘතිය සම්පූර්ණයෙන්ම විනාස කරමින් සිටින්නේ ආණ්ඩුව.'

class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))

class OfflineSummarizer(VideoSummarizer):
    def __init__(self):
        self.accept_ai_summary = False
        self.accept_english_ai_summary = True
        self.tokenizer = FakeTokenizer()
    def load_model(self, name):
        self.active_model_name = name
    def split_into_chunks(self, text, max_tokens=400):
        return [text]
    def summarize_chunk(self, text):
        return SI_CANDIDATE if self.active_model_name == self.MODEL_NAME else EN_CANDIDATE

s = OfflineSummarizer()
si = s.summarize(SI, 'si')
quotes = re.findall('“([^”]+)”', si['summary'])
assert len(quotes) == 5
assert all(q in SI for q in quotes), 'Every excerpt must be an exact source span'
assert sum(len(q.split()) for q in quotes) <= 60
for required in ('සංස්කෘතිය', 'සංඝ', 'අධ්‍යාපනය', 'පෝර ටික නෑ', 'රජයේ සේවකයෝ'):
    assert required in si['summary'], required
assert 'බේත්වියොන්ට' not in si['summary']
assert si['summary'].startswith('කථිකයාගේ ප්‍රකාශයෙන් උපුටාගත් කොටස්:')
assert si['summary_method'] == 'extractive_fallback'
assert si['ai_candidate_summary'] == SI_CANDIDATE
assert not si['negation_consistent']
assert s.sinhala_clauses('ඔවුන්ට පෝර නෑ කියලා ඔහු කියනවා.')[0] == 'ඔවුන්ට පෝර නෑ කියලා ඔහු කියනවා.'
en = s.summarize(EN, 'en')
assert en['summary'] == EN_CANDIDATE
assert en['summary_method'] == 'english_ai_heuristic_summary'
keys = {'language', 'summary', 'summary_evidence', 'ai_candidate_summary', 'summary_method', 'groundedness_score', 'coverage_score', 'negation_consistent', 'chunk_count', 'model'}
assert set(si) == set(en) == set(s.summarize('', 'si')) == keys
print('PASS: source spans, five-topic fixture, negation, attribution, English acceptance, return keys.')
print('This test does not evaluate model generation or factual accuracy.')
