import collections
import re
import time
import pandas as pd
from pathlib import Path
from itertools import islice
from datasketch import MinHash, MinHashLSH
from pympler import asizeof


def main():
    output_dir = Path('results')

    # Data downloaded from
    # https://www.kaggle.com/c/avito-duplicate-ads-detection
    df = pd.read_csv('data/duplicate-ads/ItemInfo_train.csv.zip',
                     usecols=['itemID', 'title', 'description'],
                     keep_default_na=False,
                     na_values=[])
    pairs = pd.read_csv('data/duplicate-ads/ItemPairs_train.csv.zip',
                        usecols=['itemID_1', 'itemID_2', 'isDuplicate'])
    duplicates = set([
        (min((x['itemID_1'], x['itemID_2'])), max(x['itemID_1'], x['itemID_2']))
        for _, x in pairs[pairs['isDuplicate'] == 1].iterrows()
    ])

    print(f'Read data from disk, shape = {df.shape}')

    df = df.set_index('itemID')
    all_descriptions = df['title'] + ' ' + df['description']
    
    output_dir.mkdir(parents=True, exist_ok=True)

    text_sizes = [5000, 15000, 50000, 150000, 500000]
    with open(output_dir / 'all_pairs_profiling.csv', 'w') as fp:
        write_compare_all_pairs_results(all_descriptions, duplicates, fp, text_sizes,
                                        k=7, num_perm=128, threshold=0.9)

    write_length_histograms(df, output_dir)


def write_compare_all_pairs_results(all_descriptions, groundtruth, output_file,
                                    text_sizes, k=7, num_perm=128, threshold=0.8):
    print('Profiling all pairs performance, '
          f'k={k}, num_perm={num_perm}, threshold={threshold}')
    
    data = []
    for max_texts in text_sizes:
        print(f'max_texts = {max_texts}')

        minhashes = []
        doc_subset = all_descriptions[:max_texts]
        doc_ids = set(doc_subset.index)
        groundtruth_subset = set([
            x for x in groundtruth
            if x[0] in doc_ids and x[1] in doc_ids
        ])

        t0 = time.time()
        for key, text in doc_subset.iteritems():
            minhashes.append((str(key), create_minhash(text, num_perm, k)))

        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for key, m in minhashes:
            lsh.insert(key, m)
        preparation_duration = time.time() - t0

        query_duration = 0
        predicted = set([])
        for key, m in minhashes:
            t1 = time.time()
            results = lsh.query(m)
            query_duration += time.time() - t1

            key_int = int(key)
            results = [int(x) for x in results]
            predicted.update(
                (key_int, b) if key_int < b else (b, key_int)
                for b in results if b != key_int
            )

        metrics = compute_metrics(predicted, groundtruth_subset)

        lsh_bytes = asizeof.asizeof(lsh)
        text_bytes = doc_subset.map(lambda x: len(x.encode('utf-8'))).sum()

        data.append({
            'index_size': max_texts,
            'num_query': max_texts,
            'preparation_duration': preparation_duration,
            'query_duration': query_duration,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'num_predicted': len(predicted),
            'lsh_bytes': lsh_bytes,
            'text_utf8_bytes': text_bytes,
            'k': k,
            'num_perm': num_perm,
            'threshold': threshold,
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def write_length_histograms(df, output_dir):
    df = pd.DataFrame({
        'title_length': df['title'].map(len),
        'description_length': df['description'].map(len),
    })

    df.to_csv(output_dir / 'length_histograms.csv', index=False)


def compute_metrics(predicted, expected):
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    prec = tp/(tp + fp) if tp + fp > 0 else 0
    rec = tp/(tp + fn) if tp + fn > 0 else 0
    f1 = 2*(prec * rec)/(prec + rec) if prec + rec > 0 else 0
    return {
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
    

def create_minhash(paragraph, num_perm, k_shingle):
    paragraph = re.sub(r'\s+', ' ', paragraph)
    shingles = char_shingleset(paragraph.lower(), k=k_shingle)
    m = MinHash(num_perm=num_perm)
    m.update_batch([s.encode('utf-8') for s in shingles])
    return m


def char_shingleset(text, k):
    return frozenset(''.join(x) for x in sliding_window(text, k))


def sliding_window(iterable, n):
    it = iter(iterable)
    window = collections.deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


if __name__ == '__main__':
    main()
