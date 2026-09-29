import numpy as np
import pytest

from molml.domain import domain_summary, similarity_domain


def test_training_only_threshold_and_novel_scaffold():
    train = ['c1ccccc1O', 'c1ccccc1N', 'C1CCCCC1O', 'C1CCCCC1N']
    first = similarity_domain(train, ['c1ccccc1O', 'c1ccncc1O'], n_bits=128)
    changed_query = similarity_domain(train, ['CCCC', 'CCO'], n_bits=128)
    assert first['threshold'] == changed_query['threshold']
    assert first['novel_scaffold'].tolist() == [False, True]
    assert first['nearest_train_similarity'][0] == 1.0
    assert first['in_domain'][0]
    summary = domain_summary(first)
    assert summary['count'] == 2 and summary['novel_scaffold_count'] == 1
    assert np.all(first['train_leave_one_out_similarity'] <= 1)


def test_domain_rejects_insufficient_training_data():
    with pytest.raises(ValueError):
        similarity_domain(['CCO'], ['CCN'])
