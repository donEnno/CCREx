import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable logging messages from tensorflow

import json
import argparse

from Bio import SeqIO
from ccrex import CCREx

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PATTERN_BASE_LENGTH = 11

def main():
    
    parser = argparse.ArgumentParser(description='CCREx: A deep learning model for predicting Hendecad domains.')
    parser.add_argument('-i', '--input', help='Input file in FASTA format.', required=True)
    parser.add_argument('-o', '--output', help='Output file in JSON format.', required=True)
    parser.add_argument('-t', '--threads', help='Number of threads to use.', default=-1, type=int)
    # argument for plotting

    args = parser.parse_args()

    ccrex = CCREx(n_cpu=args.threads, verbose=False)

    records = list(SeqIO.parse(args.input, 'fasta'))

    results = {}

    for record in records:
        results[record.id] = ccrex.predict(str(record.seq))

    with open(args.output, 'w') as f:
        json.dump(results, f)



if __name__ == '__main__':
    main()