import glob
# import matplotlib
import numpy as np
import os
import pickle
import tensorflow as tf
from math import ceil, floor
from scipy import signal
from scipy.io import wavfile

from chunked_file_obj import ChunkedFileObj

# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

FRAME_SIZE_SECS = 0.025
FRAME_STRIDE_SECS = 0.01
SAMPLE_RATE = 16000
FRAME_SIZE = floor(SAMPLE_RATE * FRAME_SIZE_SECS)
FRAME_STRIDE = floor(SAMPLE_RATE * FRAME_STRIDE_SECS)

FILTER_BANK_COUNT = 40
NFFT = 512
CHANNEL_COUNT = 3
FORMATTED_TRAINING = "formatted_training_non_zero.p"
FORMATTED_TESTING = "formatted_testing_non_zero.p"
TESTING_FILES = "TEST/**/**/*.WAV"
FILES = "TRAIN/**/**/*.WAV"

PHONE_SET = {'h#': 0, 'w': 1, 'ix': 2, 'dcl': 3, 's': 4, 'ah': 5, 'tcl': 6, 'ch': 7, 'n': 8, 'ae': 9, 'kcl': 10, 't': 11, 'v': 12, 'r': 13, 'f': 14, 'y': 15, 'ux': 16, 'zh': 17, 'el': 18, 'bcl': 19, 'b': 20, 'iy': 21, 'd': 22, 'ow': 23, 'nx': 24, 'epi': 25, 'm': 26, 'dx': 27, 'k': 28, 'ih': 29, 'eh': 30, 'ao': 31, 'l': 32, 'gcl': 33, 'g': 34, 'dh': 35, 'axr': 36, 'aa': 37, 'jh': 38, 'pau': 39, 'z': 40, 'ax': 41, 'ay': 42, 'sh': 43, 'q': 44, 'pcl': 45, 'p': 46, 'ax-h': 47, 'ey': 48, 'er': 49, 'uw': 50, 'ng': 51, 'en': 52, 'aw': 53, 'hv': 54, 'oy': 55, 'uh': 56, 'hh': 57, 'th': 58, 'em': 59, 'eng': 60}

assert(FRAME_SIZE >= FRAME_STRIDE)

def create_model(features, labels, mode):
    # [batch_size, 1 frame wide, 40 frequency bins, 3 values per bin
    # (freq, derivative, second derivative)]
    input_layer = tf.reshape(
        features["x"], [-1, 1, FILTER_BANK_COUNT, CHANNEL_COUNT])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=150,
        kernel_size=[1, 8],
        padding="same",
        activation=tf.nn.relu,
        name="conv",
        bias_initializer=tf.constant_initializer(0.1, tf.int64),
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=100),
    )
    tf.summary.histogram("conv1_histogram", conv1)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[1, 6], strides=2, name="pool")
    tf.summary.histogram("pool1_histogram", pool1)

    flattened = tf.reshape(pool1, [-1, 1 * 18 * 150], name="flattened")
    tf.summary.histogram("flattened_histogram", flattened)

    dense1 = tf.layers.dense(
        inputs=flattened,
        units=1000,
        activation=tf.nn.relu,
        name="dense1",
        bias_initializer=tf.constant_initializer(0.1, tf.int64),
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=200),
    )
    dropout1 = tf.layers.dropout(
        inputs=dense1,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        name="dropout1"
    )
    dense2 = tf.layers.dense(
        inputs=dropout1,
        units=1000,
        activation=tf.nn.relu,
        name="dense2",
        bias_initializer=tf.constant_initializer(0.1, tf.int64),
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=300),
    )
    dropout2 = tf.layers.dropout(
        inputs=dense2,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        name="dropout2"
    )

    tf.summary.histogram("dense1_histogram", dense1)
    tf.summary.histogram("dropout1_histogram", dropout1)
    tf.summary.histogram("dense2_histogram", dense2)
    tf.summary.histogram("dropout2_histogram", dropout2)

    # shape [-1, PHONE_SET]
    logits = tf.layers.dense(
        inputs=dropout2, units=len(PHONE_SET), name="logits")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`. Probabilities refers to what we think a particular
        # example will be.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            name="optimizer",
        )
        # labels is a [1] tensor but we'll use argmax to get its value
        correct_prediction = tf.equal(
            tf.argmax(logits, 1), labels, name="is_correct_tensor")
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name="accuracy_tensor")
        tf.summary.scalar("accuracy_summary", accuracy)

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def pre_emphasis_filter(data):
    # We pre emphasize to amplify higher frequencies.
    return np.append(data[0], data[1:] - 0.97 * data[:-1])

# Number of frames needed to include all of the data.
def get_frames_count(samples):
    if samples <= FRAME_SIZE:
        return 1
    else:
        return 1 + int(ceil((1.0 * samples - FRAME_SIZE) / FRAME_STRIDE))

def frame(data):
    # Pad
    num_frames = get_frames_count(len(data))
    pad_len = int((num_frames - 1) * FRAME_STRIDE + FRAME_SIZE)
    data = np.append(data, np.zeros(pad_len))
    # Framify
    indices = (
        np.tile(np.arange(0, int(FRAME_SIZE)), (num_frames, 1))
        + np.tile(
            np.arange(0, num_frames * FRAME_STRIDE, FRAME_STRIDE),
            (FRAME_SIZE, 1)
        ).T
    )
    frames = data[indices.astype(np.int32, copy=False)]
    return frames

def window(framed_data):
    return framed_data * np.hamming(FRAME_SIZE)

def rfft(framed_data):
    fft_magnitudes = np.absolute(np.fft.rfft(framed_data, NFFT))
    return ((1.0 / NFFT) * ((fft_magnitudes) ** 2))

def hertz_to_mel(hz):
    return (2595 * np.log10(1 + hz / 700.0))

def mel_to_hertz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)

def hertz_axis(max_freq):
    low_freq_mel = 0
    high_freq_mel = hertz_to_mel(max_freq)
    mel_axis = np.linspace(low_freq_mel, high_freq_mel, FILTER_BANK_COUNT + 2)
    hz_axis = mel_to_hertz(mel_axis)
    return hz_axis

def mel_filter_banks(framed_fft):
    nyquist_freq = SAMPLE_RATE / 2.0
    hz_axis = hertz_axis(nyquist_freq)
    bins = np.floor((NFFT + 1) * hz_axis / SAMPLE_RATE)

    fbank = np.zeros((FILTER_BANK_COUNT, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, FILTER_BANK_COUNT + 1):
        f_m_minus = int(bins[m - 1])
        f_m = int(bins[m])
        f_m_plus = int(bins[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    filter_banks = np.dot(framed_fft, fbank.T)
    filter_banks = np.where(
        filter_banks == 0, np.finfo(float).eps, filter_banks)
    return filter_banks

def mean_normalize(spectogram):
    return spectogram - (np.mean(spectogram, axis=0) + 1e-8)

def spectogram(data):
    data = pre_emphasis_filter(data)
    framed_data = frame(data)
    framed_data = window(framed_data)

    power_spectrum = rfft(framed_data)

    spect = mel_filter_banks(power_spectrum)
    # Convert to dB.
    spect = 20 * np.log10(spect)
    spect = mean_normalize(spect)

    return spect

# Returns shape [frames].
def format_training_data(file, frames):
    """Formats raw timit data to be in the proper format.
    """
    answer = np.full(frames, 0, dtype=np.int64)
    with open(file + ".PHN", "r") as fobj:
        for line in fobj:
            start, end, phoneme = line.split(" ")

            phoneme = phoneme.replace("\n", "")
            assert(phoneme in PHONE_SET)

            phoneme_id = PHONE_SET[phoneme]

            indices = np.arange(
                floor(float(start) / FRAME_STRIDE),
                min(floor(float(end) / FRAME_STRIDE) + 1, len(answer)),
            )

            answer[indices] = np.full(len(indices), phoneme_id)

    return answer

# Returns samples in shape [frames, f_bands, channels].
def format_samples(data, sample_count):
    data = data.astype(np.float16) / np.iinfo(np.dtype('int16')).min

    assert(len(data) <= sample_count)

    # Make every file have the same number of samples.
    data = np.pad(
        data,
        (0, max(0, sample_count - len(data))),
        mode="constant",
        constant_values=0,
    )

    data = spectogram(data)
    delta_data = np.gradient(data, axis=1)
    delta_delta_data = np.gradient(delta_data, axis=1)

    complete_data = np.array([data, delta_data, delta_delta_data])

    # Channels should be the last axis
    return np.moveaxis(complete_data, 0, 2)

def get_raw_stats(raw_files):
    max_samples = 0
    file_count = 0
    for file in glob.iglob(raw_files):
        _, data = wavfile.read(file)

        max_samples = max(max_samples, len(data))
        file_count += 1
    return max_samples, file_count

def format_raw_data(data_file, raw_files):
    max_samples, file_count = get_raw_stats(raw_files)

    with open(data_file, "wb") as fileobj:
        i = 0
        for file in glob.iglob(raw_files):
            # We will use WAV files as the canonical file.
            file, _ = file.split(".")
            print("Formatting", file)

            sample_rate, wfile = wavfile.read(file + ".WAV")

            assert(sample_rate == SAMPLE_RATE)

            frames = get_frames_count(max_samples)
            answer = format_training_data(file, frames)
            data = format_samples(wfile, max_samples)

            assert(len(data) == len(answer))

            pickle.dump((data, answer), fileobj)

            i += 1

def load_preformatted_data(data_file, raw_files, trim=False):
    max_samples, file_count = get_raw_stats(raw_files)
    frames = get_frames_count(max_samples)

    out_data = np.zeros(
        [frames * file_count, FILTER_BANK_COUNT, CHANNEL_COUNT],
        dtype=np.float32,
    )
    out_labels = np.zeros(
        [frames * file_count],
        dtype=np.int64,
    )
    indices = set()
    with open(data_file, "rb") as fileobj:
        pointer = 0
        for i in range(0, file_count):
            # print("Loading file", i)
            data, answer = pickle.load(fileobj)

            assert(len(data) == len(answer))

            if trim:
                trimmings = np.where(answer == 0)
                data = np.delete(data, trimmings, axis=0)
                answer = np.delete(answer, trimmings, axis=0)

            out_data[pointer : pointer + len(data)] = data
            out_labels[pointer : pointer + len(answer)] = answer
            indices.update(range(pointer, pointer + len(answer)))
            pointer += len(data)

    if trim:
        out_data = out_data[list(indices)]
        out_labels = out_labels[list(indices)]

    assert(len(out_data) == len(out_labels))

    return out_data, out_labels

def train():
    if not os.path.isfile(FORMATTED_TRAINING):
        print("Formatting training data")
        format_raw_data(FORMATTED_TRAINING, FILES)

    print("Loading formatted training data")

    training_data, training_labels = load_preformatted_data(
        FORMATTED_TRAINING, FILES, trim=True)

    assert(not np.any(np.isnan(training_data)))
    assert(not np.any(np.isnan(training_labels)))

    print(training_labels.shape, training_data.shape)

    print(
        "Percent zeros:",
        float((training_labels == 0).sum()) / len(training_labels),
    )

    print("Starting training")

    classifier = tf.estimator.Estimator(
        model_fn=create_model, model_dir="cnn_model")

    tensors_to_log = {
        "probabilities": "softmax_tensor",
        "correct": "is_correct_tensor",
        "accuracy": "accuracy_tensor",
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_data},
        y=training_labels,
        batch_size=100,
        num_epochs=10,
        shuffle=True,
    )
    classifier.train(
        input_fn=train_input_fn,
        steps=None,
        hooks=[logging_hook],
    )

def test():
    classifier = tf.estimator.Estimator(
        model_fn=create_model, model_dir="cnn_model")

    if not os.path.isfile(FORMATTED_TESTING):
        print("Formatting testing data")
        format_raw_data(FORMATTED_TESTING, TESTING_FILES)

    print("Loading formatted testing data")

    testing_data, testing_labels = load_preformatted_data(
        FORMATTED_TESTING, TESTING_FILES, trim=True)

    print(
        "Percent zeros:",
        float((testing_labels == 0).sum()) / len(testing_labels),
    )

    print(testing_labels.shape, testing_data.shape)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": testing_data},
        y=testing_labels,
        num_epochs=1,
        shuffle=False,
    )
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
