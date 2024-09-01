import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import argparse

from Bio import SeqIO
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from .ccrex import CCREx
from .utils.utils import convert_ndarray_to_list, plot_predictions

BASE_LENGTH = 55

def main():
    parser = argparse.ArgumentParser(description='Run CCREx')
    parser.add_argument('-i', '--input', required=True, help='Input fasta file path')
    parser.add_argument('-o', '--output', type=str, default='.', help='Output directory')
    parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads')
    parser.add_argument('-p', '--plot', type=bool, default=False, help='Wether to plot the results (True/False)')

    args = parser.parse_args()

    ccrex = CCREx(n_cpu=args.threads, verbose=False)

    records = list(SeqIO.parse(args.input, 'fasta'))

    results = {}

    for record in records:
        seq = str(record.seq.upper())

        if len(seq) < BASE_LENGTH:
            print(f'Sequence {record.id} is too short. Skipping...')
            continue

        results[record.id] = ccrex.predict(seq)

    results = convert_ndarray_to_list(results)

    results_path = os.path.join(args.output, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)

    if args.plot:
        plot_predictions(results, args.output)

if __name__ == '__main__':
    main()
