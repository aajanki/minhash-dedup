import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    result_dir = Path('results')
    df = pd.read_csv(result_dir / 'length_histograms.csv')
    plot_hist(df)


def plot_hist(df):
    title_lens = df['title_length']
    desc_lens = df['description_length']

    title_lens.hist()
    plt.title('title')
    plt.savefig('results/title_length_histogram.png')
    
    plt.figure()
    desc_lens.hist(bins=20)
    plt.title('description')
    plt.savefig('results/description_length_histogram.png')
    plt.show()


if __name__ == '__main__':
    main()
