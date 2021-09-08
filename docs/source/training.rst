(Re)training *jDAS*
===================

While in some cases the pretrained model provided in the GitHub repository will perform reasonably well out of the box, in practice, retraining the model is recommended, particularly in the following scenarios:

1. A new experiment was conducted (e.g. in a different location, or with different interrogator settings like the gauge length or maximum sensing distance).
2. The conditions during one experiment strongly vary. This can be the case when new noise sources are introduced (construction works on-land, microseismic noise, etc.), or when the signal-to-noise ratio significantly changes.
3. One particular event of interest, such as a major earthquake, occurs. In this case the model can be trained in "single-sample mode": instead of training on a large data set and optimising the model parameters for a (potentially) wide data range, the training is done on a very specific data range. Consequently, the *jDAS* model will try to achieve the best denoising performance for this specific data set, at the cost of generalisation.

Note that multiple models can be trained for different conditions (e.g. nighttime/daytime, on-land and submarine segments of the cable, etc.). 

The ``examples/retraining_example.ipynb`` notebook covers in detail how to prepare the data, retrain the model, and save the model state for later use. The bare-minimum procedure for retraining is as follows::

    from jDAS import JDAS
    jdas = JDAS()
    data_loader = jdas.init_dataloader(data)
    model = jdas.load_model()
    model.fit(data_loader, epochs=50)
    
Depending on the characteristics of the data, retraining can take anywhere between a minute and an hour on a standard GPU, but in most cases it is expected to take only a few minutes. When the appropriate callbacks are defined (included in the tutorial in the ``examples`` directory), the retrained model is saved to a user-defined location, and can be loaded prior to denoising as::

    model = jdas.load_model("path/to/saved-model.h5")
    clean_data = jdas.denoise(data)
