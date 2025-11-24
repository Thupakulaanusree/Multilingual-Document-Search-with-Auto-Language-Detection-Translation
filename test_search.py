from src.search_engine import MiniSearchEngine

def test_search():
    e=MiniSearchEngine()
    e.build_index('data/sample_corpus.jsonl')
    r=e.search('python')
    assert len(r)>0
