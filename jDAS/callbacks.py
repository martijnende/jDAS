import tensorflow.keras as keras


def tensorboard(logdir):
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=logdir,
        profile_batch=0,
        update_freq="epoch",
        histogram_freq=0,
    )
    return tensorboard_callback


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