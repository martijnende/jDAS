*jDAS* denoising
================

Denoising DAS data with *jDAS* is fast and easy, and can be summarised in 3 steps:

1. Instantiate the ``JDAS`` class.
2. Load a pretrained model with ``JDAS.load_model()``.
3. Call the denoising routine ``JDAS.denoise()`` with the data to be denoised.

**Load a** *jDAS* **model**

The *jDAS* repository already contains a pretrained model that is loaded by default when calling ``JDAS.load_model()`` without additional arguments. If a different model is available (e.g. one that is trained on a specific dataset), it can be loaded by pointing ``model_path`` to the directory where the frozen model is stored (e.g. ``JDAS.load_model(model_path="path/to/model/saved-model.h5")``).

**Denoise the DAS data**

The workhorse of *jDAS* is the ``denoise()`` routine. Under the hood, this routine breaks up the data into small chunks of 2048 time samples and reconstructs the data channel-by-channel, after which the junks are concatenated to obtain the *approximately* same shape as the input; since the input is broken up into an integer number of fixed-size chunks, the output for a single DAS channel will be of size ``# of chunks x 2048``.

In some cases it could happen that parasitic low-frequency artifacts are introduced, which can be filtered out with a standard bandpass filter. This post-processing can be enabled by setting ``postfilter=True`` (default: ``False``) when calling ``denoise()``. The frequency band (and temporal sampling frequency) are specified by the ``filter_band`` argument. These frequencies are given in Hertz.

Example of denoising the data, including post-process filtering in a 1-10 Hz frequency band (with a sampling frequency of 50 Hz)::

    jdas = JDAS()
    model = jdas.load_model()
    clean_data = jdas.denoise(noisy_data, postfilter=True, filter_band=(1, 10, 50))

Note that when calling ``JDAS.denoise`` for the first time in a session, Keras/TensorFlow will rebuild and optimise the model, which takes some time (of the order of 10 seconds). Once this initiation step is done, the subsequent calls to the model are very fast.

**Data requirements**

The DAS data are assumed to be organised in a 2D matrix, for which each row represents one DAS channel (point along the fibre) and each column represents one time sample. So for ``Nch`` channels and ``Nt`` time samples, the data shape is ``(Nch, Nt)``. Moreover, the data are assumed to be Euclidean, meaning that the spacing between each sample is constant in both space and time (constant gauge length and time-sampling frequency).

The pretrained model was trained on a dataset that was bandpass filtered in a 1-10 Hz pass band, and was sampled at 50 Hz with a gauge length (and channel spacing) of 19.2 m. For data with a different gauge length, retraining of the model is required. However, retraining is not necessary if the ratio of the frequency pass band to the Nyquist frequency remains fixed. To give an example, a pass band of 1-10 Hz sampled at 50 Hz is the same as a pass band of 5-50 Hz sampled at 250 Hz. However, since retraining is relatively fast, it is recommended to do so regardless.

A last requirement is that the DAS data be approximately normalised before being passed onto the neural network. This is automatically taken care of by the ``denoise()`` routine, but it is something to keep in mind when calling the trained model directly.
