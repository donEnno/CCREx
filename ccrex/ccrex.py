import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from keras.models import load_model                                         # type: ignore

from .utils.utils import one_hot_encoder, get_kmers, geometric_mean

class CCREx:

    def __init__(self, n_cpu=1, verbose=False):
        self.n_cpu = n_cpu
        self.verbose = verbose

        self.kmer_len = 55   

        # Build CCREx
        self.models = self._setup_models()

    def _setup_models(self):
        models_basepath = '/home/enno/uni/SS24/CCREx_dev/models/simple_models/AVILM_R5_M1/'  # 'ccrex/models/'
        models = os.listdir(models_basepath)

        models = [load_model(models_basepath + model) for model in models]

        return models

    def predict(self, sequence):
        result = {'hen': [], 'hep': []}

        seq = sequence.upper()

        encoded_seq = np.array(one_hot_encoder([seq]))[0]
        kmers, positions = get_kmers(encoded_seq, self.kmer_len)
        
        hen_predictions = []
        hep_predictions = []

        for model in self.models:
            preds = model.predict(kmers, verbose=0)
            temp_hen_predictions = []
            temp_hep_predictions = []

            for j in range(len(seq)):
                relevant_indices = [idx for idx, pos in enumerate(positions) if pos <= j < pos + self.kmer_len]
                relevant_preds = preds[relevant_indices]

                if len(relevant_preds) > 0:
                    temp_hen_predictions.append(geometric_mean(relevant_preds[:, 1]))
                    temp_hep_predictions.append(geometric_mean(relevant_preds[:, 2]))
                else:
                    temp_hen_predictions.append(np.nan)
                    temp_hep_predictions.append(np.nan)

            hen_predictions.append(temp_hen_predictions)
            hep_predictions.append(temp_hep_predictions)

        hen_predictions = np.nanmean(hen_predictions, axis=0)
        hep_predictions = np.nanmean(hep_predictions, axis=0)

        result['hen'] = hen_predictions
        result['hep'] = hep_predictions

        return result
