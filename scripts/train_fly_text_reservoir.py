"""Train a supervised text readout on the published fixed MaleCNS circuit.

Run prepare_inputs.py first. The label-free text encoder is fitted on training
text only. The deployed classifier receives only simulated circuit states.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from prepare_inputs import ORIGINAL_MODEL_SHA

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)?"
MAX_TOKENS, POOLS, LEAK, SEED = 96, 4, 0.35, 42


def log(message):
    print(message, flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize(text):
    return re.findall(TOKEN_PATTERN, text.lower().replace('\u2019', "'"))[:MAX_TOKENS]


def load_circuit(folder):
    meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
    raw = (folder / 'model.bin').read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_MODEL_SHA:
        raise ValueError('Original circuit checksum differs from the published one.')
    arrays = {}
    for name in ('indptr', 'indices', 'weights'):
        spec = meta['model']['arrays'][name]
        arrays[name] = np.frombuffer(raw, dtype=spec['dtype'], count=spec['count'], offset=spec['offset']).copy()
    n = meta['neurons']
    assert n == 499 and len(arrays['weights']) == 15865 and arrays['indptr'][-1] == 15865
    # Rows are POSTSYNAPTIC neurons; column indices are PRESYNAPTIC sources.
    matrix = sparse.csr_matrix((arrays['weights'], arrays['indices'], arrays['indptr']), shape=(n, n))
    assert np.isfinite(matrix.data).all() and np.max(matrix.indices) < n
    assert connected_components(matrix, directed=True, connection='strong', return_labels=False) == 1
    return matrix, meta


def load_splits(folder):
    splits, seen, removed, counts = {}, set(), {}, {}
    for name in ('train', 'validation', 'test'):
        rows = pq.read_table(folder / (name + '.parquet')).to_pylist()
        texts, labels, dropped = [], [], 0
        for row in rows:
            key = ' '.join(tokenize(row['text']))
            if not key or key in seen:
                dropped += 1
                continue
            seen.add(key)
            texts.append(row['text'])
            labels.append(int(row['label']))
        y = np.asarray(labels)
        assert set(y) == set(range(6))
        splits[name] = (texts, y)
        removed[name] = dropped
        counts[name] = np.bincount(y, minlength=6).tolist()
    return splits, {'removed_empty_or_duplicate': removed, 'class_counts': counts,
                    'policy': 'Official splits; canonical duplicates removed with train then validation then test precedence. No author IDs available.'}


def fit_encoder(train_texts, dim):
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 min_df=2, max_features=12000, sublinear_tf=True, dtype=np.float32)
    tfidf = vectorizer.fit_transform(train_texts)
    svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=SEED).fit(tfidf)
    embedding = svd.components_.T.copy()
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)
    embedding = np.vstack([np.zeros((1, dim), dtype=np.float32), embedding]).astype(np.float32)
    vocabulary = {word: int(index) + 1 for word, index in vectorizer.vocabulary_.items()}
    return vectorizer, tfidf, embedding, vocabulary


def encode(text, vocabulary):
    seq = [vocabulary.get(token, 0) for token in tokenize(text)] or [0]
    return seq + [seq[-1]] * max(0, POOLS - len(seq))


def extract(sequences, matrix, projected, embedding, batch_size=128):
    """Length-batched sparse recurrence; temporal pooling excludes padding."""
    n, d = matrix.shape[0], embedding.shape[1]
    out = np.zeros((len(sequences), POOLS * n), dtype=np.float32)
    direct = np.zeros((len(sequences), POOLS * d), dtype=np.float32)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        lengths = np.array([len(sequences[i]) for i in indices])
        tokens = np.zeros((len(indices), int(lengths.max())), dtype=np.int32)
        for row, index in enumerate(indices):
            tokens[row, :lengths[row]] = sequences[index]
        state = np.zeros((len(indices), n), dtype=np.float32)
        pooled = np.zeros((len(indices), POOLS, n), dtype=np.float32)
        inputs = np.zeros((len(indices), POOLS, d), dtype=np.float32)
        for t in range(tokens.shape[1]):
            state = (1 - LEAK) * state + LEAK * np.tanh(matrix.dot(state.T).T + projected[tokens[:, t]])
            for part in range(POOLS):
                active = (t >= (part * lengths // POOLS)) & (t < ((part + 1) * lengths // POOLS))
                pooled[active, part] += state[active]
                inputs[active, part] += embedding[tokens[active, t]]
        for part in range(POOLS):
            count = ((part + 1) * lengths // POOLS - part * lengths // POOLS)[:, None]
            pooled[:, part] /= count
            inputs[:, part] /= count
        out[indices] = pooled.reshape(len(indices), -1)
        direct[indices] = inputs.reshape(len(indices), -1)
    return out, direct


def scrambled(matrix):
    rng, csc = np.random.default_rng(SEED + 1), matrix.tocsc()
    row, col, values = [], [], []
    for source in range(matrix.shape[0]):
        weights = csc.data[csc.indptr[source]:csc.indptr[source+1]]
        row.extend(rng.choice(matrix.shape[0], len(weights), replace=False))
        col.extend([source] * len(weights))
        values.extend(weights)
    result = sparse.csr_matrix((values, (row, col)), shape=matrix.shape, dtype=np.float32)
    radius = float(np.max(np.abs(np.linalg.eigvals(result.toarray()))))
    result *= 0.9 / radius
    return result


def metrics(y, scores):
    pred = np.argmax(scores, axis=1)
    return {'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, average='macro')),
            'examples': len(y), 'confusion_matrix': confusion_matrix(y, pred, labels=np.arange(6)).tolist(),
            'per_class': classification_report(y, pred, target_names=LABELS, output_dict=True, zero_division=0)}


def fit_readout(name, features, splits, alphas=(1.0, 10.0, 100.0, 1000.0)):
    scaler = StandardScaler(with_mean=not sparse.issparse(features['train']))
    train = scaler.fit_transform(features['train'].astype(np.float64))
    validation = scaler.transform(features['validation'].astype(np.float64))
    trials, best = [], None
    for alpha in alphas:
        clf = RidgeClassifier(alpha=alpha, class_weight='balanced', solver='lsqr' if sparse.issparse(train) else 'cholesky')
        clf.fit(train, splits['train'][1])
        value = float(f1_score(splits['validation'][1], clf.predict(validation), average='macro'))
        trials.append({'alpha': alpha, 'validation_macro_f1': value})
        if best is None or value > best[0]:
            best = (value, clf)
    clf = best[1]
    scores = clf.decision_function(scaler.transform(features['test'].astype(np.float64)))
    result = {'selected_alpha': float(clf.alpha), 'validation_trials': trials, 'test': metrics(splits['test'][1], scores)}
    log(f'{name}: test accuracy {result["test"]["accuracy"]:.4f}, macro-F1 {result["test"]["macro_f1"]:.4f}')
    return result, clf, scaler, scores


def simulate_one(text, matrix, projected, embedding, vocabulary, coef, bias, silence=()):
    sequence = encode(text, vocabulary)
    mask = np.ones(matrix.shape[0], dtype=np.float32)
    mask[list(silence)] = 0
    state = np.zeros(matrix.shape[0], dtype=np.float32)
    states = []
    for token in sequence:
        state = ((1 - LEAK) * state + LEAK * np.tanh(matrix @ state + projected[token])) * mask
        states.append(state.copy())
    states = np.asarray(states)
    pooled = np.concatenate([states[p * len(states) // POOLS:(p + 1) * len(states) // POOLS].mean(axis=0) for p in range(POOLS)])
    return pooled @ coef.T + bias, states


def export(folder, matrix, projected, vocabulary, clf, scaler, meta, report):
    folder.mkdir(parents=True, exist_ok=True)
    coef = clf.coef_.astype(np.float64) / scaler.scale_
    bias = clf.intercept_.astype(np.float64) - coef @ scaler.mean_
    ranked = np.argsort(-np.linalg.norm(coef.reshape(6, POOLS, 499), axis=(0, 1))).astype('<u2')
    arrays = {'indptr': matrix.indptr.astype('<u4'), 'indices': matrix.indices.astype('<u2'),
              'weights': matrix.data.astype('<f4'), 'token_drive': projected.astype('<f4'),
              'coef': coef.astype('<f8'), 'bias': bias.astype('<f8'), 'ranked': ranked}
    blob, specs = bytearray(), {}
    for key, array in arrays.items():
        blob.extend(b'\0' * ((-len(blob)) % 8))
        specs[key] = {'offset': len(blob), 'count': array.size, 'shape': list(array.shape), 'dtype': array.dtype.str}
        blob.extend(array.tobytes())
    (folder / 'weights.bin').write_bytes(blob)
    raw_anatomy = np.fromfile(ROOT / 'data/oruk/anatomy.bin', dtype='<i2').reshape(-1, 4)
    owner = raw_anatomy[:, 3].view('<u2')
    keep = []
    for neuron in range(499):
        vertices = np.flatnonzero(owner == neuron)
        assert len(vertices) % 2 == 0
        keep.extend(vertices.reshape(-1, 2)[::8].ravel())
    anatomy = raw_anatomy[np.asarray(keep)].copy()
    (folder / 'anatomy.bin').write_bytes(anatomy.tobytes())
    manifest = {'schema': 1, 'name': 'MaleCNS text reservoir', 'labels': LABELS, 'neurons': 499,
                'edges': matrix.nnz, 'leak': LEAK, 'pools': POOLS, 'max_tokens': MAX_TOKENS,
                'vocabulary': vocabulary, 'arrays': specs, 'weights_sha256': hashlib.sha256(blob).hexdigest(),
                'anatomy_sha256': sha(folder / 'anatomy.bin'), 'geometry': {'vertices': len(anatomy), 'stride_bytes': 8, 'position_scale': 2},
                'centers': meta['geometry']['centers'], 'report': report,
                'cell_identifiers': 'Circuit-local indices 0..498. Original MaleCNS body IDs are not supplied by this public export.',
                'graph_source_sha256': ORIGINAL_MODEL_SHA,
                'readout_inputs': 'Four temporal means of fly states only. No direct text branch.',
                'encoder': 'Label-free training-text TF-IDF/LSA token embeddings and seeded fixed random projection.',
                'source': 'https://oruk.ai/research/we-taught-a-fruit-fly-to-read-human-emotion',
                'data_source': 'https://male-cns.janelia.org/download/'}
    (folder / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')), encoding='utf-8')
    (folder / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return coef, bias, ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dim', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    begin = time.time()
    with threadpool_limits(limits=args.threads):
        matrix, meta = load_circuit(ROOT / 'data/oruk')
        log(f'Verified {matrix.shape[0]} neurons, {matrix.nnz} connections, one strongly connected component.')
        splits, split_report = load_splits(ROOT / 'data/emotion')
        log('Retained split counts: ' + str({key: len(value[0]) for key, value in splits.items()}))
        vectorizer, train_tfidf, embedding, vocabulary = fit_encoder(splits['train'][0], args.input_dim)
        input_weights = np.random.default_rng(SEED).normal(0, 0.3, (499, args.input_dim)).astype(np.float32)
        projected = (embedding @ input_weights.T).astype(np.float32)
        sequences = {key: [encode(text, vocabulary) for text in val[0]] for key, val in splits.items()}
        report = {'labels': LABELS, 'seed': SEED, 'split': split_report, 'input_dim': args.input_dim,
                  'vocabulary_size': len(vocabulary), 'models': {}, 'sources': json.loads((ROOT / 'data/sources.json').read_text())}
        fitted, direct, test_scores = {}, {}, {}
        for name, graph in [('fly', matrix), ('scrambled', scrambled(matrix)), ('no_recurrence', sparse.csr_matrix(matrix.shape, dtype=np.float32))]:
            log(f'Computing {name} states...')
            features = {}
            for split, seq in sequences.items():
                features[split], direct[split] = extract(seq, graph, projected, embedding)
                log(f'  {name}/{split}: {len(seq)} sentences')
            result, clf, scaler, scores = fit_readout(name, features, splits)
            report['models'][name], test_scores[name] = result, scores
            if name == 'fly':
                fitted = {'clf': clf, 'scaler': scaler}
        result, _, _, scores = fit_readout('text_encoder_only', direct, splits)
        report['models']['text_encoder_only'], test_scores['text_encoder_only'] = result, scores
        tfidf_features = {key: train_tfidf if key == 'train' else vectorizer.transform(value[0]) for key, value in splits.items()}
        result, _, _, _ = fit_readout('tfidf_text_only', tfidf_features, splits, (1.0, 10.0, 100.0))
        report['models']['tfidf_text_only'] = result
        rng, truth, differences = np.random.default_rng(SEED), splits['test'][1], []
        for _ in range(500):
            sample = rng.integers(0, len(truth), len(truth))
            fly = f1_score(truth[sample], test_scores['fly'][sample].argmax(axis=1), average='macro', labels=np.arange(6), zero_division=0)
            control = f1_score(truth[sample], test_scores['scrambled'][sample].argmax(axis=1), average='macro', labels=np.arange(6), zero_division=0)
            differences.append(fly-control)
        report['fly_minus_scrambled_macro_f1_ci95'] = np.quantile(differences, [.025, .975]).tolist()
        report['majority_accuracy'] = float(np.max(np.bincount(truth)) / len(truth))
        report['limitations'] = ['One fixed circuit, one random seed; intervals are conditional on this split.',
                                 'English social-media data with noisy distant labels; no neutral category.',
                                 'No author metadata, so author leakage cannot be ruled out.',
                                 'Only the readout uses labels. Simplified dynamics and input are engineered.',
                                 'Raw ridge scores are not probabilities. Research/educational dataset use.']
        report['seconds'] = time.time() - begin
        folder = ROOT / 'dist/model'
        coef, bias, ranked = export(folder, matrix, projected, vocabulary, fitted['clf'], fitted['scaler'], meta, report)
        verification = []
        for text in ['I feel happy and excited about tomorrow.', 'I feel miserable and lonely.', 'I feel scared and anxious.', 'I am furious about this.', 'I feel loved and cared for.', 'I feel surprised and amazed.', 'why?', 'zzzxqv', '<script>alert(1)</script>']:
            scores, states = simulate_one(text, matrix, projected, embedding, vocabulary, coef, bias)
            muted, _ = simulate_one(text, matrix, projected, embedding, vocabulary, coef, bias, ranked[:50])
            verification.append({'text': text, 'scores': scores.tolist(), 'silenced_50_scores': muted.tolist(),
                                 'last_state': states[-1].tolist(), 'mean_abs_activity': np.abs(states).mean(axis=1).tolist()})
        (folder / 'verification.json').write_text(json.dumps(verification), encoding='utf-8')
        (ROOT / 'research/results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        log(f'Exported trained model and anatomy to {folder}; elapsed {time.time()-begin:.1f}s')


if __name__ == '__main__':
    main()
