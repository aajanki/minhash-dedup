import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    plt.rcParams["figure.figsize"] = (9.6, 7.2)
    plt.rcParams["font.size"] = 16
    plt.rcParams["lines.linewidth"] = 2
    
    result_dir = Path('results')

    df_all_pairs = (
        pd.read_csv(result_dir / 'all_pairs_profiling.csv')
        .rename(columns={
            'num_query': 'Number of documents',
            'preparation_duration': 'Index construction duration',
            'query_duration': 'Query duration',
            'lsh_bytes': 'MinHash LSH index',
            'text_utf8_bytes': 'Raw document text',
        })
    )
    df_all_pairs['Raw document text'] = df_all_pairs['Raw document text']/(1024*1024)
    df_all_pairs['MinHash LSH index'] = df_all_pairs['MinHash LSH index']/(1024*1024)

    # Query duration should be approximately linear.
    df_all_pairs.plot(
        x='Number of documents',
        y='Query duration',
        title='Duration of finding all duplicated pairs',
        ylabel='Seconds',
        style='.-',
        markersize=10,
        legend=False,
    )
    plt.savefig('results/query_duration.png')

    df_all_pairs.plot(
        x='Number of documents',
        y='Index construction duration',
        title='Index preparation duration',
        ylabel='Seconds',
        color='r',
        style='.-',
        markersize=10,
        legend=False,
    )
    plt.savefig('results/index_duration.png')

    # Size of the LSH index as a function of the input text size
    # (UTF-8 encoded bytes)
    df_all_pairs.plot(
        x='Raw document text',
        y='MinHash LSH index',
        title='Memory consumption (megabytes)',
        ylabel='MinHash LSH index',
        style='.-',
        markersize=10,
        legend=False,
    )
    plt.savefig('results/memory.png')

    # Accuracy of the detected duplicated pairs
    df_all_pairs.plot(
        x='Number of documents',
        y=['precision', 'recall', 'f1'],
        style='.-',
        markersize=10,
    )
    plt.savefig('results/metrics.png')

    plt.show()


if __name__ == '__main__':
    main()
