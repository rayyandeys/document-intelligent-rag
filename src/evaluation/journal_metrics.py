"""Document-level metrics and paired percentile-bootstrap intervals."""
import math
import numpy as np


def metrics(ranking, relevance, k=10):
    if len(set(ranking)) != len(ranking):
        raise ValueError('Rankings must contain unique document IDs')
    relevant = {d for d, grade in relevance.items() if grade > 0}
    if not relevant:
        raise ValueError('Query has no positive relevance judgments')
    gains = [max(0, relevance.get(d, 0)) for d in ranking[:k]]
    # Linear relevance gains, matching trec_eval ndcg_cut convention.
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((relevance[d] for d in relevant), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    result = {f'hit_at_{n}': float(any(d in relevant for d in ranking[:n])) for n in (1, 3, 5, 10)}
    result.update(mrr_at_10=next((1 / i for i, d in enumerate(ranking[:10], 1) if d in relevant), 0.0),
                  ndcg_at_10=dcg/idcg, recall_at_10=sum(d in relevant for d in ranking[:10])/len(relevant))
    return result


def interval(values, seed=42, samples=10000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite per-query values')
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 256):
        count = min(256, samples-start)
        means[start:start+count] = values[rng.integers(len(values), size=(count,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), ci95_low=float(np.quantile(means,.025)), ci95_high=float(np.quantile(means,.975)))
