import pandas as pd

splits = {'validation': 'validation.jsonl.gz', 'test': 'test.jsonl.gz'}
df = pd.read_json("hf://datasets/mteb/scidocs-reranking/" + splits["validation"], lines=True)
df.to_csv('scidocs_validation.csv', index=False)
