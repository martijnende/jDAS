import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

""" Setting random seeds """
seed = 42

# TensorFlow
tf.random.set_seed(seed)

# Python
import random as python_random
python_random.seed(seed)

# NumPy (random number generator used for sampling operations)
rng = np.random.default_rng(seed)


class DataLoader(keras.utils.Sequence):

    def __init__(self, X, batch_size=16, batch_multiplier=10):
        
        # Data matrix
        self.X = X
        # Number of samples
        self.N_samples = X.shape[0]
        # Number of stations
        self.Nx = X.shape[1]
        # Number of time sampling points
        self.Nt = X.shape[2]
        # Number of time sampling points for an input sample (fixed by architecture)
        self.win = 2048
        # Number of stations per batch sample
        self.N_sub = 11
        # Starting indices of the slices
        self.station_inds = np.arange(self.Nx - self.N_sub)
        # Batch size
        self.batch_size = batch_size
        self.batch_multiplier = batch_multiplier

        self.on_epoch_end()

    def __len__(self):
        """ Number of mini-batches per epoch """
        return int(self.batch_multiplier * self.N_samples * self.Nx / float(self.batch_size * self.N_sub))

    def on_epoch_end(self):
        """ Modify data """
        self.__data_generation()
        pass

    def __getitem__(self, idx):
        """ Select a mini-batch """
        batch_size = self.batch_size
        selection = slice(idx * batch_size, (idx + 1) * batch_size)
        samples = np.expand_dims(self.samples[selection], -1)
        masked_samples = np.expand_dims(self.masked_samples[selection], -1)
        masks = np.expand_dims(self.masks[selection], -1)
        return (samples, masks), masked_samples

    def __data_generation(self):
        """ Generate a total batch """
        
        # Number of mini-batches
        N_batch = self.__len__()
        N_total = N_batch * self.batch_size
        # Buffer for mini-batches
        samples = np.zeros((N_total, self.N_sub, self.win))
        # Buffer for masks
        masks = np.ones_like(samples)
        
        batch_inds = np.arange(N_total)
        np.random.shuffle(batch_inds)
        
        # Number of subsamples to create
        n_mini = N_total // self.N_samples
        
        # Loop over samples
        for s, sample in enumerate(self.X):
            # Random selection of station indices
            selection = rng.choice(self.station_inds, size=n_mini, replace=False)
            # Random time slice
            t_start = rng.integers(low=0, high=self.Nt-self.win)
            t_slice = slice(t_start, t_start + self.win)
            # Time reversal
            order = rng.integers(low=0, high=2) * 2 - 1
            sign = rng.integers(low=0, high=2) * 2 - 1
            # Loop over station indices
            for k, station in enumerate(selection):
                # Selection of stations
                station_slice = slice(station, station + self.N_sub)
                subsample = sign * sample[station_slice, t_slice][:, ::order]
                # Get random index of this batch sample
                batch_ind = batch_inds[s * n_mini + k]
                # Store waveforms
                samples[batch_ind] = subsample
                # Select one waveform to blank
                blank_ind = rng.integers(low=0, high=self.N_sub)
                # Create mask
                masks[batch_ind, blank_ind] = 0
                
        self.samples = samples
        self.masks = masks
        self.masked_samples = samples * (1 - masks)
        pass
