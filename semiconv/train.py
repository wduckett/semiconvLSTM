import argparse
import tensorflow as tf
import numpy as np
from semiconv.SemiConvolutionalLSTM import SemiConvLSTM

def create_shifted_frames(data):
    x = data[:, 0: data.shape[1] - 1, :, :]
    y = data[:, data.shape[-1], :, :]
    return x, y

def customMAE(y_true, y_pred):
    y_mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    y_useful = y_pred * y_mask
    maskToArr = tf.cast((tf.size(y_mask) / tf.cast(tf.reduce_sum(y_mask), tf.int32)), tf.float32)
    return tf.keras.losses.mean_absolute_error(y_true, y_useful) * maskToArr

def main(args):
    # turns GPU off as training on the GPU was not fast on my computer
    tf.config.set_visible_devices([], 'GPU')

    print("loading data..")
    data = np.load(args.data_path)
    data = tf.reshape(data, [-1, 168, 32, 32, 1])

    data = np.log(data + 1)
    maxVal = int(tf.reduce_max(data))
    data = data / (maxVal * 1.2)

    data = data[:int(data.shape[0] * 0.9), ...]

    val_dataset = data[int(data.shape[0]*0.45):int(data.shape[0]*0.55)]

    train_dataset = tf.concat([data[:int(data.shape[0]*0.45)], data[int(data.shape[0]*0.55):int(data.shape[0]*0.9)]], axis=0)

    train_dataset = tf.random.shuffle(train_dataset, seed=1000)
    val_dataset = tf.random.shuffle(val_dataset, seed=1000)

    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)

    inp = tf.keras.layers.Input(shape=(None, *x_train.shape[2:]))

    print("running model...")
    
    x = SemiConvLSTM(
        filters=32,
        rank=2,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="tanh",
    )(inp)

    x = tf.keras.layers.BatchNormalization()(x)

    x = SemiConvLSTM(
        filters=32,
        rank=2,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="tanh",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = SemiConvLSTM(
        filters=1,
        rank=2,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=False,
        activation="sigmoid",
    )(x)

    model = tf.keras.models.Model(inp, x)
    model.compile(
        loss=customMAE,
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate),
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
    csvLogger = tf.keras.callbacks.CSVLogger(
        args.log_file,
        separator=',',
        append=False
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        args.checkpoint_path,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    )

    print("fitting model...")

    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr, csvLogger, model_checkpoint_callback],
    )

    model.save(args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a SemiConvLSTM model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the processed data file (npy format).')
    parser.add_argument('--log_file', type=str, default='result.csv', help='Path to the CSV log file.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.model.keras', help='Path to the model checkpoint file.')
    parser.add_argument('--model_path', type=str, default='model.tf', help='Path to save the final model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    args = parser.parse_args()
    main(args)


