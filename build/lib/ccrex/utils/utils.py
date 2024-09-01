import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt

default_colors = ['#648FFF', '#DC267F', '#FFB000', '#FE6100', '#785EF0', '#6D9843', '#2B5858', '#3D1B5A', '#98D4D8', '#C8DE41']
grey = '#545454'
plt.rcParams['axes.prop_cycle'] = cycler(color=default_colors)
mpl.rcParams['scatter.edgecolors'] = 'black'

font = {'family' : 'monospace',
          'weight' : 'bold',
          'size'   : 16}
mpl.rc('font', **font)  


AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


def one_hot_encoder(input_seqs):
    """
    One-hot encodes input sequences.

    Args:
        input_seqs (list): List of input sequences.

    Returns:
        numpy.ndarray: One-hot encoded input sequences.
    """

    unique_letters = sorted(set(AMINO_ACIDS))
    letter_to_index = {letter: index for index, letter in enumerate(unique_letters)}
    
    encodings = []

    for seq in input_seqs:
        one_hot_matrix = np.zeros((len(seq), len(unique_letters)))

        for i, letter in enumerate(seq):
            if letter not in AMINO_ACIDS:
                continue
                
            one_hot_matrix[i, letter_to_index[letter]] = 1
            
        encodings.append(one_hot_matrix)

    return encodings


def get_kmers(seq, len_kmer):
    kmers = []
    positions = []

    for j in range(0, len(seq) - len_kmer + 1):
        kmers.append(seq[j:j+len_kmer]) 
        positions.append(j)

    last_kmer = seq[-len_kmer:]
    last_pos = len(seq) - len_kmer

    if np.all(last_kmer == kmers[-1]):
        kmers.append(last_kmer)
        positions.append(last_pos)
    
    kmers = np.array(kmers) 
    positions = np.array(positions)
    
    return kmers, positions


def geometric_mean(arr):
    return np.prod(arr)**(1.0/len(arr))


def convert_ndarray_to_list(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, dict):
            convert_ndarray_to_list(value)
    return d

def prob_to_pred(pred, t=0.5):
    return np.array([1 if x > t else 0 for x in pred])

def plot_predictions(predictions, path, t=0.5):  
        """
        Plot the predictions on the given sequence.

        Parameters:
            predictions (dict): Dictionary containing the predictions.
            sequence (str): Protein sequence to plot the predictions on.
            save (bool): Flag to save the plot.
        """

        plot_dir = path + '/ccrex_plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        for key, value in predictions.items():
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))

            ax.set_ylim([-0.05, 1.05])

            ax.plot(value['hen'], label='HEN', color=default_colors[0])
            ax.plot(value['hep'], label='HEP', color=default_colors[1])

            ax.axhline(y=t, color=grey, linestyle='--', label='Threshold')

            ax.set_title(key)
            ax.set_xlabel('Position')
            ax.set_ylabel('Probability')
            ax.legend()

            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{key}.png')
            plt.close()
