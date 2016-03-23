import time
import numpy as np
import theano
import theano.tensor as T
import lasagne


def dict_to_array(ref_keys, test_keys, distance_matrix):
    """
    Convert dictionary to numpy array.
    :param ref_keys: list with reference keys
    :param test_keys: list with test keys
    :param distance_matrix: dictionary representing distance matrix
    :return: numpy array
    """
    rows = len(test_keys)
    cols = len(ref_keys)
    matrix = np.zeros((rows, cols), dtype=np.float64)

    for i, test in enumerate(test_keys):
        for j, ref in enumerate(ref_keys):
            matrix[i][j] = distance_matrix[ref, test]

    return matrix


def get_targets(keys, references):
    """
    Get numeric values for keys.
    :param keys: list with keys
    :param references: dictionary with references
    :return: numpy array
    """
    targets = []

    for key in keys:
        targets.append(references[key])

    return np.asarray(targets, dtype=np.int32)


def targets_to_vectors(class_count, targets):
    """
    Transform integer targets to target vectors.
    :param class_count: number of output classes
    :param targets: numpy array (vector) with targets
    :return: numpy array
    """
    matrix = np.zeros((len(targets), class_count), dtype=np.int32)

    for i, label in enumerate(targets):
        matrix[i, label] = 1

    return matrix


def prepare_dataset(ref_keys, train_keys, val_keys, test_keys, distance_matrix, references):
    """
    Prepare dataset for neural network training and testing.
    :param ref_keys: list with reference keys
    :param train_keys: list with training keys
    :param val_keys:  list with validation keys
    :param test_keys: list with test keys
    :param distance_matrix: dictionary representing distance matrix
    :param references: dictionary with references
    :return: numpy arrays
    """
    X_train = dict_to_array(ref_keys, train_keys, distance_matrix)
    X_val = dict_to_array(ref_keys, val_keys, distance_matrix)
    X_test = dict_to_array(ref_keys, test_keys, distance_matrix)

    train_targets = get_targets(train_keys, references)
    val_targets = get_targets(val_keys, references)
    test_targets = get_targets(test_keys, references)

    class_count = len(ref_keys)

    y_train = targets_to_vectors(class_count, train_targets)
    y_val = targets_to_vectors(class_count, val_targets)
    y_test = targets_to_vectors(class_count, test_targets)

    print('{0: <10}{1: >10}.{2: <15}{3: <10}{4: >10}.{5}'.format('X_train:', X_train.shape, X_train.dtype,
                                                                 'y_train:', y_train.shape, y_train.dtype))
    print('{0: <10}{1: >10}.{2: <15}{3: <10}{4: >10}.{5}'.format('X_val:', X_val.shape, X_val.dtype,
                                                                 'y_val:', y_val.shape, y_val.dtype))
    print('{0: <10}{1: >10}.{2: <15}{3: <10}{4: >10}.{5}'.format('X_test:', X_test.shape, X_test.dtype,
                                                                 'y_test:', y_test.shape, y_test.dtype))
    print ''

    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """
    Yield input and target minibatch.
    :param inputs: numpy array with network input data
    :param targets: numpy array with network targets
    :param batch_size: integer
    :param shuffle: true/false
    :return: iterator
    """
    assert len(inputs) == len(targets)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def build_mlp(input_dim, output_dim, depth, num_units, drop_input=None, drop_hidden=None, input_var=None):
    """
    Build custom MLP.
    :param input_dim: count of input units
    :param output_dim: count of output units
    :param depth: hidden layers count
    :param num_units: count of units in hidden layers
    :param drop_input: input dropout value
    :param drop_hidden: hidden dropout value
    :param input_var: Theano variable
    :return: lasagne network
    """
    network = lasagne.layers.InputLayer(shape=(None, input_dim), input_var=input_var)

    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)

    nonlin = lasagne.nonlinearities.rectify

    for _ in range(depth):
        network = lasagne.layers.DenseLayer(network, num_units, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)

    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, output_dim, nonlinearity=softmax)

    return network


def train(num_epochs, batch_size, X_train, y_train, X_val, y_val, input_dim, output_dim, depth, num_units,
          drop_input=None, drop_hidden=None, report=50):
    """
    Train neural network.
    :param num_epochs: training epochs count
    :param batch_size: integer
    :param X_train: numpy array with train data
    :param y_train: numpy array with train targets
    :param X_val: numpy array with validation data
    :param y_val: numpy arrays with validation targets
    :param input_dim: count of input units
    :param output_dim: count of output units
    :param depth: hidden layers count
    :param num_units: count of units in hidden layers
    :param drop_input: input dropout value
    :param drop_hidden: hidden dropout value
    :param report: report output frequency
    :return: lasagne network
    """
    input_var = T.dmatrix('inputs')
    target_var = T.imatrix('targets')

    network = build_mlp(input_dim, output_dim, depth, num_units, drop_input, drop_hidden, input_var)

    # create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # create a loss expression for validation with deterministic forward pass (disable dropout layers)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    # create an expression for the classification accuracy
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)

    # compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # compile a function computing the validation loss and accuracy
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    for epoch in range(num_epochs):
        # full pass over the training data
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # full pass over the validation data
        val_err = 0
        val_acc = 0
        val_batches = 0

        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        if (epoch + 1) % report == 0:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            print("\ttraining loss:\t\t\t{:.6f}".format(train_err / train_batches))
            print("\tvalidation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("\tvalidation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    return network


def recognize(X_test, network):
    """
    Recognize test data.
    :param X_test: numpy array with test data
    :param network: lasagne network
    :return: list with recognized data
    """
    recognition = []

    for row in xrange(len(X_test)):
        prob = np.array(lasagne.layers.get_output(network, X_test[row], deterministic=True).eval())
        recognition.append(np.argmax(prob))

    return recognition


def compute_accuracy(y_test, recognition):
    """
    Compute recognition accuracy.
    :param y_test: numpy array with test targets
    :param recognition: list with recognized data
    :return: accuracy
    """
    correct = 0
    wrong = 0

    for i in xrange(len(y_test)):
        if recognition[i] == np.argmax(y_test[i]):
            correct += 1
        else:
            wrong += 1

    accuracy = 100.0 * correct / float(correct + wrong)

    return accuracy


def recognize_and_compute_accuracy(X_test, y_test, network):
    """
    Compute recognition accuracy.
    :param X_test: numpy array with test data
    :param y_test: numpy array with test targets
    :param network: lasagne network
    :return: accuracy
    """
    correct = 0

    for row in xrange(len(y_test)):
        prob = np.array(lasagne.layers.get_output(network, X_test[row], deterministic=True).eval())

        if np.argmax(prob) == np.argmax(y_test[row]):
            correct += 1

    return 100.0 * correct / len(y_test)


def dump_info(file_name, depth, num_units, drop_input=None, drop_hidden=None):
    """
    Save network specification to .txt file.
    :param file_name: model filename
    :param depth: hidden layers count
    :param num_units: count of units in hidden layers
    :param drop_input: input dropout value
    :param drop_hidden: hidden dropout value
    :return: nothing
    """
    with open(file_name, 'w') as fw:
        fw.write('{0: <15}{1}\n'.format('depth:', depth))
        fw.write('{0: <15}{1}\n'.format('num_units:', num_units))
        fw.write('{0: <15}{1}\n'.format('drop_input:', drop_input))
        fw.write('{0: <15}{1}\n'.format('drop_hidden:', drop_hidden))
