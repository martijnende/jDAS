import sys
import os
import keras
import numpy as np
import gc

from .filters import taper_filter
from .dataloader import DataLoader
from .callbacks import tensorboard, checkpoint

# from ..models.jDAS_arch import CallBacks, DataGenerator, UNet

cwd = os.path.dirname(__file__)

# __version__ = "0.1.0"
# __author__ = "Martijn van den Ende"


class JDAS:
    
    __version__ = "0.1.0"
    __author__ = "Martijn van den Ende"
    
    def __init__(self):
        self.taper_filter = taper_filter
        self.callback_tensorboard = tensorboard
        self.callback_checkpoint = checkpoint
        pass
    
    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(cwd, "..", "models", "pretrained_model.h5")
            
        self.model = keras.models.load_model(model_path)
        return self.model
    
    
    def init_dataloader(self, data, batch_size=16, batch_multiplier=10):
        dataloader = DataLoader(data, batch_size=batch_size, batch_multiplier=batch_multiplier)
        return dataloader
    
    
    @staticmethod
    def _J_construct(model, data, Nsub):
        """Prepare sample and masks, and subsequently perform J-invariant filtering

        Parameters
        ----------

        data : `numpy.array`
            DAS data
        Nsub : int
            Number of DAS channels to be combined into one model input sample

        Returns
        -------

        rec : `numpy.array`
            J-invariant filtered sample

        """

        Nch, Nt = data.shape

        masks = np.ones((Nch, Nsub, Nt, 1))
        eval_samples = np.zeros_like(masks)

        gutter = Nsub // 2
        mid = Nsub // 2

        for i in range(gutter):
            masks[i, i] = 0
            eval_samples[i, :, :, 0] = data[:Nsub]

        for i in range(gutter, Nch - gutter):
            start = i - mid
            stop = i + mid + 1

            masks[i, mid] = 0
            eval_samples[i, :, :, 0] = data[start:stop]

        for i in range(Nch - gutter, Nch):
            masks[i, i - Nch] = 0
            eval_samples[i, :, :, 0] = data[-Nsub:]

        result = model.predict((eval_samples, masks))
        rec = np.sum(result, axis=1)[:, :, 0]
        _ = gc.collect()

        return rec


    def denoise(self, data, postfilter=False, filter_band=(1, 10, 50), verbose=True):
        """Filter data with a J-invariant model

        This function automatically splits the input `data` into suitable samples to feed into the J-invariant model for filtering.

        Parameters
        ----------

        data : `numpy.array`
            DAS data
        postfilter : bool, default False
            Whether or not to bandpass filter after J-invariant denoising
        filter_band : tuple
            The pass band and sampling frequency (in Hertz) used for the bandpass filtering. Only used when `postfilter==True`.
            Order is (low freq, high freq, sampling freq)
        verbose : bool, default True
            Whether or not to output the denoising progress

        Returns
        -------

        recs_all : `numpy.array`
            J-invariant filtered data

        """
        
        model = self.model
        
        if model is None:
            print("No model has been loaded!")
            return None

        Nch, Nt_tot = data.shape
        Nsub = 11
        Nt = 2048
        Nsamples = Nt_tot // Nt
        
        if Nch < Nsub:
            print(f"Data size too small along spatial-axis: {Nch} < {Nsub}")
            return
        if Nt_tot < Nt:
            print(f"Data size too small along time-axis: {Nt_tot} < {Nt}")
            return
        
        # Normalise data
        scale = data.std()
        data_scaled = data / scale
        
        # If there is only 1 chunk: make single call
        if Nt_tot == Nt:
            recs_all = self._J_construct(model, data_scaled, Nsub)
        
        # If we have more than 1 chunk
        else:
            # Buffer for output data
            recs_all = np.zeros((Nch, Nsamples * Nt))
            
            # Output time steps
            n_out = np.arange(0, Nsamples, max(Nsamples // 10, 1))
            
            # Loop over chunks
            for n in range(Nsamples):
                
                # Check for verbose output
                if (n in n_out) and verbose:
                    print(f"Processing {n+1} / {Nsamples}")
                
                # Data range for this chunk
                t_slice = slice(n * Nt, (n+1) * Nt)
                
                # Get J-invariant reconstruction for this chunk
                rec = self._J_construct(model, data_scaled[:, t_slice], Nsub)
                
                # Add to output buffer
                recs_all[:, t_slice] = rec
        
        # Do post-process bandpass filtering (if requested)
        if postfilter:
            fmin, fmax, samp = filter_band
            recs_all = self.taper_filter(recs_all, fmin, fmax, samp)
        
        # Rescale data
        recs_all *= scale
        del data_scaled

        return recs_all
