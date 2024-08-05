import argparse
import tensorflow as tf
import numpy as np
from semiconv.SemiConvolutionalLSTM import SemiConvLSTM

tf.config.set_visible_devices([], 'GPU')

def customMAE(y_true, y_pred):
    maxVal = 7 # this is the maximum value of the dataset the model was trained on
    y_true = y_true[:, -1, ...]
    y_pred = y_pred[:, -1, ...]
    y_true = y_true * (maxVal * 1.2)
    y_pred = y_pred * (maxVal * 1.2)
    y_true = tf.exp(y_true) - 1 
    y_pred = tf.exp(y_pred) -1
    y_mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    y_useful = y_pred * y_mask
    maskToArr = tf.cast((tf.size(y_mask) / tf.cast(tf.reduce_sum(y_mask), tf.int32)), tf.float32)
    return tf.keras.losses.mean_absolute_error(y_true, y_useful) * maskToArr

def main(args):
    data = np.load(args.data_path)
    data = tf.reshape(data, [-1, 168, 32, 32, 1])
    data = data[-1, ...]
    data = tf.reshape(data, [1, 168, 32, 32, 1])
    data = np.log(data + 1)
    maxVal = int(tf.reduce_max(data))
    data = data / (7 * 1.2) # 7 is the maximum value of the training set for this model and so has to be used

    sensorMask = np.load(args.sensor_mask_path)
    sensorMask = sensorMask / 255
    sensorMask = tf.reshape(sensorMask, [32, 32, 1])

    model = tf.keras.models.load_model(args.model_path, custom_objects={"SemiConvLSTM": SemiConvLSTM, "customLoss": customMAE})

    predictionArr = []
    denormedPredictions = []
    for i in range(24):
        prediction = model.predict(data)
        prediction = prediction[-1, -1, ...] * sensorMask
        denormd = np.flip(prediction.numpy(), axis=0) * (maxVal * 1.2)
        denormd = np.exp(denormd) - 1
        denormedPredictions.append(denormd)
        predictionArr.append(np.flip(prediction.numpy(), axis=0))
        prediction = tf.reshape(prediction, [1, 1, 32, 32, 1])
        data = tf.concat([data, prediction], axis=1)

    np.save(args.output_path, denormedPredictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run traffic prediction using a pretrained SemiConvLSTM model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file (npy format).')
    parser.add_argument('--sensor_mask_path', type=str, required=True, help='Path to the sensor mask file (npy format).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the prediction output (npy format).')

    args = parser.parse_args()
    main(args)
