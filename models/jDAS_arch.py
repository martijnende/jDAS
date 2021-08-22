import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import BatchNormalization, UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.optimizers import Adam

""" Setting random seeds """
seed = 42

# TensorFlow
tf.random.set_seed(seed)

# Python
import random as python_random
python_random.seed(seed)

# NumPy (random number generator used for sampling operations)
rng = np.random.default_rng(seed)


class DataGenerator(keras.utils.Sequence):

    def __init__(self, X, N_sub=10, batch_size=16, batch_multiplier=10):
        
        # Data matrix
        self.X = X
        # Number of samples
        self.N_samples = X.shape[0]
        # Number of stations
        self.Nx = X.shape[1]
        # Number of time sampling points
        self.Nt = X.shape[2]
        # Number of stations per batch sample
        self.N_sub = N_sub
        # Starting indices of the slices
        self.station_inds = np.arange(self.Nx - N_sub)
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
        samples = np.zeros((N_total, self.N_sub, self.Nt))
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
            # Time reversal
            order = rng.integers(low=0, high=2) * 2 - 1
            sign = rng.integers(low=0, high=2) * 2 - 1
            # Loop over station indices
            for k, station in enumerate(selection):
                # Selection of stations
                station_slice = slice(station, station + self.N_sub)
                subsample = sign * sample[station_slice, ::order]
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

    def generate_masks(self, samples):
        """ Generate masks and masked samples """
        N_masks = self.N_masks
        N_patch = self.N_patch
        Ny = samples.shape[2]
        patch_inds = self.patch_inds
        patch_radius = self.patch_radius
        # Tile samples
        samples = np.tile(samples, [N_masks, 1, 1])
        # Add extra dimension
        samples = np.expand_dims(samples, -1)
        # Shuffle samples
        inds = np.arange(samples.shape[0])
        np.random.shuffle(inds)
        samples = samples[inds]
        # Generate complementary masks (patch = 1)
        c_masks = np.zeros_like(samples)
        for n in range(c_masks.shape[0]):
            selection = rng.choice(patch_inds, size=N_patch, replace=False)
            for sel in selection:
                i = sel // Ny
                j = sel % Ny
                slice_x = slice(i - patch_radius[0], i + patch_radius[0])
                slice_y = slice(j - patch_radius[1], j + patch_radius[1])
                c_masks[n, slice_x, slice_y] = 1
        # Masks (patch = 0)
        masks = 1 - c_masks
        # Masked samples (for loss function)
        masked_samples = c_masks * samples
        return samples, masked_samples, masks


class CallBacks:

    @staticmethod
    def tensorboard(logdir):
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=logdir,
            profile_batch=0,
            update_freq="epoch",
            histogram_freq=0,
        )
        return tensorboard_callback

    @staticmethod
    def checkpoint(savefile):
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            savefile,
            verbose=0,
            save_weights_only=False,
            save_best_only=True,
            monitor="val_loss",
            mode="auto",
            update_freq="epoch",
        )
        return checkpoint_callback


class UNet:

    def __init__(self):
        self.kernel = (3, 5)
        self.f0 = 2
        self.N_blocks = 4
        self.use_bn = True
        self.use_dropout = True
        self.dropout_rate = 0.1
        self.AA = False
        self.LR = 5e-4
        self.initializer = keras.initializers.Orthogonal()
        self.activation = tf.keras.activations.swish
        self.data_shape = (10, 1024, 1)
        pass

    def set_params(self, params):
        """
        Update model parameters
        """
        self.__dict__.update(params)
        pass

    def conv_layer(self, x, filters, kernel_size,
                   use_bn=False, use_dropout=False, activ=None):
        """
        Convolution layer > batch normalisation > activation > dropout
        """
        use_bias = True
        if use_bn:
            use_bias = False

        x = Conv2D(
            filters=filters, kernel_size=kernel_size, padding="same",
            activation=None, kernel_initializer=self.initializer,
            use_bias=use_bias
        )(x)

        if use_bn:
            x = BatchNormalization()(x)

        if activ is not None:
            x = Activation(activ)(x)

        if use_dropout:
            x = GaussianDropout(self.dropout_rate)(x)

        return x
    
    def MaxBlurPool(self, x, kernel_size=(1, 4)):
        
        if kernel_size[1] == 1:
            a = np.array([1.,])
        elif kernel_size[1] == 2:
            a = np.array([1., 1.])
        elif kernel_size[1] == 3:
            a = np.array([1., 2., 1.])
        elif kernel_size[1] == 4:    
            a = np.array([1., 3., 3., 1.])
        elif kernel_size[1] == 5:    
            a = np.array([1., 4., 6., 4., 1.])
        elif kernel_size[1] == 6:    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif kernel_size[1] == 7:    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        
        a = a / a.sum()
        a = np.repeat(a, x.shape[-1]*x.shape[-1])
        a = a.reshape((kernel_size[0], kernel_size[1], x.shape[-1], x.shape[-1]))
        
        x = MaxPool2D(pool_size=kernel_size, strides=(1, 1))(x)
        x = tf.nn.conv2d(input=x, filters=a, strides=kernel_size, padding="SAME")
        
        return x

    def construct(self):
        """
        Construct UNet model
        """

        f = self.f0
        kernel = self.kernel
        use_bn = self.use_bn
        use_dropout = self.use_dropout
        AA = self.AA
        activation = self.activation
        data_shape = self.data_shape

        input = Input(data_shape)
        mask_input = Input(data_shape)
        c_mask_input = 1 - mask_input
        x = mask_input * input

        """ Encoder """
        x = self.conv_layer(x, filters=f, kernel_size=kernel,
                            use_bn=use_bn, use_dropout=use_dropout,
                            activ=activation)
        x = self.conv_layer(x, filters=f, kernel_size=kernel,
                            use_bn=use_bn, use_dropout=use_dropout,
                            activ=activation)

        x_prev = [x]

        for i in range(self.N_blocks):
            
            if AA:
                x = self.MaxBlurPool(x, kernel_size=(1, 4))
            else:
                x = MaxPool2D(pool_size=(1, 4))(x)
            
            f = f * 2
            x = self.conv_layer(x, filters=f, kernel_size=kernel,
                                use_bn=use_bn, use_dropout=use_dropout,
                                activ=activation)
            x = self.conv_layer(x, filters=f, kernel_size=kernel,
                                use_bn=use_bn, use_dropout=use_dropout,
                                activ=activation)
            x_prev.append(x)

        """ Decoder """
        for i in range(self.N_blocks-1):
            x = UpSampling2D(size=(1, 4), interpolation="bilinear")(x)
            f = f // 2
            x = concatenate([x, x_prev[-(i+2)]])
            x = self.conv_layer(x, filters=f, kernel_size=kernel,
                                use_bn=use_bn, use_dropout=use_dropout,
                                activ=activation)
            x = self.conv_layer(x, filters=f, kernel_size=kernel,
                                use_bn=use_bn, use_dropout=use_dropout,
                                activ=activation)

        x = UpSampling2D(size=(1, 4), interpolation="bilinear")(x)  # 128
        f = f // 2
        x = concatenate([x, x_prev[0]])
        x = self.conv_layer(x, filters=f, kernel_size=kernel,
                            use_bn=use_bn, use_dropout=use_dropout,
                            activ=activation)
        x = self.conv_layer(x, filters=f, kernel_size=kernel,
                            use_bn=use_bn, use_dropout=use_dropout,
                            activ=activation)
        x = self.conv_layer(x, filters=1, kernel_size=kernel,
                            use_bn=False, use_dropout=False, activ=None)

        x = c_mask_input * x
        model = Model([input, mask_input], x)
        model.build(input_shape=[data_shape, data_shape])

        # Build and generate a summary
        # model.summary()

        # Train auto-encoder
        model.compile(
            optimizer=Adam(learning_rate=self.LR),
            loss="mean_squared_error",
        )

        return model
