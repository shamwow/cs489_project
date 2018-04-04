import glob
import numpy as np
import os
import pickle
import python_speech_features as psf
import tensorflow as tf
from fast_predict import FastPredict
from math import ceil, floor
from scipy import signal
from scipy.io import wavfile

tf.logging.set_verbosity(tf.logging.INFO)

FRAME_SIZE_SECS = 0.025
FRAME_STRIDE_SECS = 0.01
SAMPLE_RATE = 16000
FRAME_SIZE = floor(SAMPLE_RATE * FRAME_SIZE_SECS)
FRAME_STRIDE = floor(SAMPLE_RATE * FRAME_STRIDE_SECS)
MAX_FRAMES_PER_FILE = 1000

FILTER_BANK_COUNT = 40
NFFT = 512
CHANNEL_COUNT = 3
FORMATTED_TRAINING = "formatted_training.p"
FORMATTED_TESTING = "formatted_testing.p"
TESTING_FILES = "TEST/**/**/*.WAV"
FILES = "TRAIN/**/**/*.WAV"
ACCURACY_MAP_FILE = "accuracy_map.p"

LIMITED_PHONE_SET = {'h#': 0, 'w': 1, 'ix': 2, 's': 3, 'ah': 4, 'ch': 5,
'n': 6, 'ae': 7, 't': 8, 'v': 9, 'r': 10, 'f': 11, 'y': 12, 'zh': 13,
'b': 14, 'iy': 15, 'd': 16, 'ow': 17, 'm': 18, 'dx': 19, 'k': 20, 'ih': 21,
'eh': 22, 'ao': 23, 'l': 24, 'g': 25, 'dh': 26, 'aa': 27, 'jh': 28, 'z': 29,
'axr': 30, 'ax': 31, 'ay': 32, 'sh': 33, 'p': 34, 'ey': 35, 'er': 36, 'uw': 37,
'ng': 38, 'aw': 39, 'oy': 40, 'uh': 41, 'hh': 42, 'th': 43}
PHONE_SET = {
'h#': 0, 'w': 1, 'ix': 2, 's': 4, 'ah': 5, 'ch': 7, 'n': 8,
'ae': 9, 't': 11, 'v': 12, 'r': 13, 'f': 14, 'y': 15,
'zh': 17, 'b': 20, 'iy': 21, 'd': 22, 'ow': 23,
'm': 26, 'dx': 27, 'k': 28, 'ih': 29, 'eh': 30, 'ao': 31, 'l': 32, 'g': 34,
'dh': 35, 'aa': 37, 'jh': 38, 'z': 40, 'axr': 36,
'ax': 41, 'ay': 42, 'sh': 43,  'p': 46,  'ey': 48,
'er': 49, 'uw': 50, 'ng': 51, 'aw': 53,  'oy': 55, 'uh': 56,
'hh': 57, 'th': 58,
'dcl': 3, 'tcl': 6, 'kcl': 10, 'bcl': 19, 'epi': 25, 'gcl': 33,  'ax-h': 47,
'pau': 39, 'pcl': 45, 'eng': 60, 'em': 59, 'en': 52, 'nx': 24, 'ux': 16,
'el': 18, 'q': 44, 'hv': 54
}

assert(FRAME_SIZE >= FRAME_STRIDE)

def get_frames_count(samples: int) -> int:
    """Returns the number of frames needed to store `samples` samples.
    """
    if samples <= FRAME_SIZE:
        return 1
    else:
        return 1 + int(ceil((1.0 * samples - FRAME_SIZE) / FRAME_STRIDE))

# Returns samples in shape [frames, f_bands, channels].
def get_spectrogram_data(wfile):
    data = wfile.astype(np.float16) / np.iinfo(np.dtype('int16')).min

    data = psf.logfbank(
        data,
        nfft=NFFT,
        nfilt=FILTER_BANK_COUNT,
        samplerate=SAMPLE_RATE,
        winstep=FRAME_STRIDE_SECS,
        winlen=FRAME_SIZE_SECS,
    )
    delta_data = psf.delta(data, 2)
    delta_delta_data = psf.delta(delta_data, 2)

    assert(len(data) == get_frames_count(len(wfile)))

    complete_data = np.array([data, delta_data, delta_delta_data])

    # Channels should be the last axis
    return np.moveaxis(complete_data, 0, 2)

def format_training_data(file: str, training_data):
    """Formats raw timit data to be a mapping of frame to phone.

    Returns tuple: (array of phones, removed frame indices).
    """
    # Faster to allocate a large array and remove at the end.
    answer = np.full(len(training_data), -1, dtype=np.int64)
    with open(file + ".PHN", "r") as fobj:
        for line in fobj:
            start, end, phoneme = line.split(" ")

            phoneme = phoneme.replace("\n", "")
            assert(phoneme in PHONE_SET)

            if phoneme not in PHONE_SET:
                continue

            phoneme_id = PHONE_SET[phoneme]

            start_idx = get_frames_count(int(start) + 1) - 1
            end_idx = get_frames_count(int(end)) - 1
            assert(end_idx < len(training_data))
            indices = np.arange(start_idx, end_idx + 1)

            # # Remove 1 frame from each end to focus only on main part of
            # # phone. If the phone only takes up 1 or two frames, ignore it.
            # if len(indices) <= 2:
            #     continue

            # assert(len(indices) > 2)
            # indices = indices[1:-1]

            answer[indices] = np.full(len(indices), phoneme_id)

    removed = np.where(answer == -1)[0]
    answer = np.delete(answer, removed, axis=0)
    training_data = np.delete(training_data, removed, axis=0)

    assert(len(answer) == len(training_data))

    return answer, training_data

def get_raw_stats(raw_files):
    max_samples = 0
    file_count = 0
    for file in glob.iglob(raw_files):
        _, data = wavfile.read(file)

        max_samples = max(max_samples, len(data))
        file_count += 1
    return max_samples, file_count

def format_raw_data(data_file, raw_files):
    with open(data_file, "wb") as fileobj:
        for file in glob.iglob(raw_files):
            file, _ = file.split(".")
            print("Formatting", file)

            sample_rate, wfile = wavfile.read(file + ".WAV")

            assert(sample_rate == SAMPLE_RATE)

            spect = get_spectrogram_data(wfile)

            training_labels, training_data = format_training_data(file, spect)

            assert(len(training_data) == len(training_labels))

            pickle.dump((training_data, training_labels), fileobj)

def load_preformatted_data(data_file, raw_files, trim=False):
    _, file_count = get_raw_stats(raw_files)

    out_data = np.zeros(
        [MAX_FRAMES_PER_FILE * file_count, FILTER_BANK_COUNT, CHANNEL_COUNT],
        dtype=np.float32,
    )
    out_labels = np.zeros(
        [MAX_FRAMES_PER_FILE * file_count],
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

    out_data = out_data[list(indices)]
    out_labels = out_labels[list(indices)]

    assert(len(out_data) == len(out_labels))

    return out_data, out_labels

def conv_layer(inp, filters, kernel_size, name):
    with tf.name_scope(name):
        initer = tf.contrib.layers.xavier_initializer()
        w = tf.Variable(
            initer(list(kernel_size) + [CHANNEL_COUNT, 150])
        )
        b = tf.Variable(tf.constant(0.1, shape=[150]))

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)

        conv = tf.nn.conv2d(
            inp, w, strides=[1, 1, 1, 1], padding="SAME", name=name)
        act = tf.nn.relu(conv + b)
        return act

def dense_layer(inp, input_dim, neurons, name):
    with tf.name_scope(name):
        initer = tf.contrib.layers.xavier_initializer()
        w = tf.Variable(initer([input_dim, neurons]))
        b = tf.Variable(tf.constant(0.1, shape=[neurons]))

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)

        act = tf.nn.relu(tf.matmul(inp, w) + b)
        return act

def create_model(features, labels, mode):
    input_layer = tf.reshape(
        features["x"], [-1, 1, FILTER_BANK_COUNT, CHANNEL_COUNT])
    # Convolutional Layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=150,
        kernel_size=[1, 6],
        padding="same",
        activation=tf.nn.relu,
        name="conv",
    )
    tf.summary.histogram("conv1_histogram", conv1)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[1, 6], strides=2, name="pool")
    tf.summary.histogram("pool1_histogram", pool1)

    flattened = tf.reshape(pool1, [-1, 1 * 18 * 150], name="flattened")
    tf.summary.histogram("flattened_histogram", flattened)

    dense1 = tf.layers.dense(
        inputs=flattened,
        units=100,
        activation=tf.nn.relu,
        name="dense1",
    )
    dense2 = tf.layers.dense(
        inputs=dense1,
        units=100,
        activation=tf.nn.relu,
        name="dense2",
    )
    dropout = tf.layers.dropout(
        inputs=dense2,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        name="dropout"
    )

    tf.summary.histogram("dense1_histogram", dense1)
    tf.summary.histogram("dropout_histogram", dropout)
    tf.summary.histogram("dense2_histogram", dense2)

    logits = tf.layers.dense(
        inputs=dropout, units=2, name="logits")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`. Probabilities refers to what we think a particular
        # example will be.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                "output": tf.estimator.export.PredictOutput(predictions),
            },
        )

    tf.identity(labels, name="label_tensor")

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            name="optimizer",
        )
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

def train(phone):
    if not os.path.isfile(FORMATTED_TRAINING):
        print("Formatting training data")
        format_raw_data(FORMATTED_TRAINING, FILES)

    print("Loading formatted training data")

    training_data, training_labels = load_preformatted_data(
        FORMATTED_TRAINING, FILES)

    assert(not np.any(np.isnan(training_data)))
    assert(not np.any(np.isnan(training_labels)))

    print("Starting training")

    tensors_to_log = {
        "probabilities": "softmax_tensor",
        "correct": "is_correct_tensor",
        "accuracy": "accuracy_tensor",
        "label": "label_tensor",
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    phone_id = PHONE_SET[phone]
    positive_indices = np.where(training_labels == phone_id)[0]
    negative_indices = np.where(training_labels != phone_id)[0]
    training_labels[np.arange(0, len(training_labels))] = 0
    training_labels[positive_indices] = 1
    np.random.shuffle(negative_indices)

    # Remove negative indices so that ~25% of the data has negative labels.
    diff = len(negative_indices) - floor(len(positive_indices) * 1.0/3.0)
    if diff > 0:
        np.random.shuffle(negative_indices)
        remove = negative_indices[:diff]
        training_labels = np.delete(training_labels, remove, axis=0)
        training_data = np.delete(training_data, remove, axis=0)

    assert(training_labels.sum() == ceil(len(training_labels) * .75))
    assert(len(training_labels) == len(training_data))

    classifier = tf.estimator.Estimator(
        model_fn=create_model, model_dir="cnn_model_" + phone)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_data},
        y=training_labels,
        batch_size=1,
        num_epochs=1,
        shuffle=True,
    )
    classifier.train(
        input_fn=train_input_fn,
        steps=None,
        hooks=[logging_hook],
    )

def complete_test():
    if not os.path.isfile(FORMATTED_TESTING):
        print("Formatting testing data")
        format_raw_data(FORMATTED_TESTING, TESTING_FILES)

    print("Loading formatted testing data")

    testing_data, testing_labels = load_preformatted_data(
        FORMATTED_TESTING, TESTING_FILES)

    classifiers = {}
    for phone in PHONE_SET:
        classifier = tf.estimator.Estimator(
            model_fn=create_model, model_dir="cnn_model_" + phone)
        print(classifier.get_variable_names())
        classifiers[phone] = FastPredict(classifier)

    accuracy_map = None
    with open(ACCURACY_MAP_FILE, "rb") as fileobj:
        accuracy_map = pickle.load(fileobj)

    total_guessed = 0
    total_correct = 0
    for i, (data, label) in enumerate(zip(testing_data, testing_labels)):
        max_certainty = 0
        predicted_phone = None
        for phone in PHONE_SET:
            input_features = {"x": np.array([data])}
            prediction = classifiers[phone].predict(input_features)[0]
            is_phone = prediction["classes"] == 1
            _, certainty = prediction["probabilities"]

            stats = accuracy_map[phone]
            normalized = certainty * stats["accuracy"] - stats["loss"]

            if is_phone and normalized > max_certainty:
                predicted_phone = phone
                max_certainty = normalized

        if predicted_phone is not None:
            total_correct += int(PHONE_SET[predicted_phone] == label)
            total_guessed += 1

        if i % 1000 == 0:
            print("Total guessed", total_guessed)
            print("Total correct", total_correct)

    print("Accuracy:", float(total_correct) / len(testing_labels))

def test(phone):
    if not os.path.isfile(FORMATTED_TESTING):
        print("Formatting testing data")
        format_raw_data(FORMATTED_TESTING, TESTING_FILES)

    print("Loading formatted testing data")

    testing_data, testing_labels = load_preformatted_data(
        FORMATTED_TESTING, TESTING_FILES)

    phone_id = PHONE_SET[phone]
    negative_indices = np.where(testing_labels != phone_id)[0]
    positive_indices = np.where(testing_labels == phone_id)[0]
    testing_labels[np.arange(0, len(testing_labels))] = 0
    testing_labels[positive_indices] = 1

    # Remove negative indices so the number of positive and negative indices
    # is the same.
    diff = len(negative_indices) - len(positive_indices)
    if diff > 0:
        np.random.shuffle(negative_indices)
        remove = negative_indices[:diff]
        testing_labels = np.delete(testing_labels, remove, axis=0)
        testing_data = np.delete(testing_data, remove, axis=0)

    classifier = tf.estimator.Estimator(
        model_fn=create_model, model_dir="cnn_model_" + phone)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": testing_data},
        y=testing_labels,
        num_epochs=1,
        shuffle=False,
    )
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    return eval_results

def train_all():
    for phone in PHONE_SET:
        train(phone)

def test_all():
    for phone in PHONE_SET:
        test(phone)

def tune(phone):
    accuracy_map = None
    with open(ACCURACY_MAP_FILE, "rb") as fileobj:
        accuracy_map = pickle.load(fileobj)

    while accuracy_map[phone]["accuracy"] < .75:
        print("Tuning", phone)
        train(phone)
        res = test(phone)
        accuracy_map[phone] = res
        with open(ACCURACY_MAP_FILE, "wb") as fileobj:
            pickle.dump(accuracy_map, fileobj)

def tune_all():
    for phone in PHONE_SET:
        tune(phone)


complete_test()

