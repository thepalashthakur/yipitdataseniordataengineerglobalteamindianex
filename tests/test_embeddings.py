import json

import numpy as np

from yipit_pipeline.embeddings import top_similar_articles


def test_top_similarity_excludes_self_and_returns_bridge():
    article_ids = ["A", "B", "C", "D"]
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    json_rows, bridge = top_similar_articles(article_ids, matrix, top_n=3)
    assert len(json_rows) == 4
    assert len(bridge) == 12
    for article_id, raw_ids in zip(article_ids, json_rows):
        similar = json.loads(raw_ids)
        assert len(similar) == 3
        assert len(set(similar)) == 3
        assert article_id not in similar

