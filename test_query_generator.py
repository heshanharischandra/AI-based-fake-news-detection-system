"""Run from the project root: python test_query_generator.py."""
import json
from copy import deepcopy
from app.services.claim_analyzer import ClaimAnalyzer
from app.services.query_generator import QueryGenerator


def main():
    analyzer = ClaimAnalyzer()
    generator = QueryGenerator()
    si = analyzer.analyze('මේ රටේ සංස්කෘතිය විනාස කරමින් ඉන්නවා. ගොවියොන්ට පෝර ටික නෑ', 'si')['claims']
    assert len(si) == 2
    assert all(c['review_screening_status'] == 'heuristic_screened' for c in si)
    original = deepcopy(si)
    enriched = generator.enrich_claims(si, 'si')
    assert si == original
    assert enriched[0]['search_queries']['status'] == 'draft_needs_review'
    assert enriched[1]['search_queries']['status'] == 'draft_screened'
    assert enriched[1]['search_queries']['queries'][0]['query'] == si[1]['text']
    assert 'නෑ' in enriched[1]['search_queries']['queries'][0]['query']

    en_text = 'The hospital reported that 12 patients were not admitted on Saturday.'
    en = analyzer.analyze(en_text, 'en')['claims']
    assert len(en) == 1
    assert en[0]['review_screening_status'] == 'not_performed'
    draft = generator.enrich_claims(en, 'en')[0]['search_queries']
    assert draft['status'] == 'draft_unscreened'
    assert draft['queries'][0]['query'] == en_text
    assert len(draft['queries']) == 2
    for q in draft['queries']:
        assert all(w in q['query'] for w in ('reported', '12', 'not', 'Saturday'))
        assert set(q['query'].split()) <= set(en_text.split())

    # Entity labels are not injected into source queries: NER can be wrong.
    item = dict(en[0], entities={'locations': ['InventedLocation']}, start=1.0, end=4.0)
    enriched_item = generator.enrich_claims([item], 'en')[0]
    assert enriched_item['entities'] == item['entities']
    assert enriched_item['start'] == 1.0 and enriched_item['end'] == 4.0
    assert all('InventedLocation' not in q['query'] for q in enriched_item['search_queries']['queries'])
    restored = json.loads(json.dumps(enriched_item, ensure_ascii=False))
    assert restored == enriched_item

    # Old JSON without screening metadata must default to unscreened.
    assert generator.generate({'text': 'A source statement', 'needs_review': False}, 'en')['status'] == 'draft_unscreened'
    assert generator.generate({'text': ''}, 'si')['queries'] == []
    assert generator.enrich_claims([], 'si') == []

    for text in (
        'ගොවියොන්ට පෝර ටික නෑ කියලා රජයේ සේවකයෝ කියනවා.',
        'ගොවියොන්ට පෝර ටික නෑ නම් රජයේ සේවකයෝ උදව් කරනවා.',
        'ගොවියොන්ට පෝර ටික නෑ රජයේ සේවකයෝ අමාරුවෙන් ජීවත් වෙන්නේ',
    ):
        claims = analyzer.analyze(text, 'si')['claims']
        result = generator.enrich_claims(claims, 'si')
        for claim in result:
            assert claim['search_queries']['queries'][0]['query'] == claim['text']
    assert analyzer.analyze('ඔයා වියදම් අඩු කරලා සල්ලි ඉතුරු කරන්න.', 'si')['claim_count'] == 0

    print('PASS: review states, source wording, negation, quantities, reporting, metadata, JSON, and unchanged advice filtering.')
    print('No searches were sent. Drafts do not establish truth or search readiness.')
    print('Sinhala query example:', enriched[1]['search_queries']['queries'][0]['query'])
    print('English status:', draft['status'])


if __name__ == '__main__':
    main()
