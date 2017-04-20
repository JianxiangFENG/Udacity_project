import helper
import problem_unittests as tests

import numpy as np
import pickle
import tensorflow as tf

cifar10_dataset_folder_path = 'cifar-10-batches-py'

def normalize(x):
    max_val = np.amax(x)
    min_val = np.amin(x)
    return (np.array(x)-min_val)/(max_val - min_val)
#tests.test_normalize(normalize)

def one_hot_encode(x):
    return np.eye(10)[x]
#tests.test_one_hot_encode(one_hot_encode)

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """
    Load a batch of the dataset
    """
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    features = normalize(features)
    labels = one_hot_encode(labels)
    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        validation_count = int(len(features) * 0.1)

        # Prprocess and save a batch of training data
        _preprocess_and_save(
            normalize,
            one_hot_encode,
            features[:-validation_count],
            labels[:-validation_count],
            'preprocess_batch_' + str(batch_i) + '.p')

        # Use a portion of training batch for validation
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation.p')

    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the training data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all training data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_training.p')

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

def get_input(image_shape, n_classes):
    x = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')
    keep_pro = tf.placeholder(tf.float32, name='keep_pro')
    return x, y, keep_pro

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    x_shape = x_tensor.get_shape().as_list()
    weights = tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], x_shape[3], conv_num_outputs], mean=0.0, stddev=0.1))
    biases = tf.Variable(tf.random_normal([conv_num_outputs]))
    #convolution
    conv_result = tf.nn.conv2d(x_tensor, weights, strides=[1,conv_strides[0],conv_strides[1],1], padding='SAME')
    conv_result = tf.nn.bias_add(conv_result, biases)
    conv_result = tf.nn.relu(conv_result)
    #max-pooling
    pool_result = tf.nn.max_pool(conv_result, ksize=[1,pool_ksize[0],pool_ksize[1],1], strides=[1,pool_strides[0],pool_strides[1],1], padding='SAME')
    return pool_result

def flatten(x_tensor):
    return tf.contrib.layers.flatten(x_tensor)

def fully_conn(x_tensor, num_outputs):
    weights = tf.Variable(tf.random_normal([x_tensor.get_shape().as_list()[1],num_outputs], mean=0.0, stddev=0.1))
    bias = tf.Variable(tf.random_normal([num_outputs]))
    return tf.nn.relu(tf.add(tf.matmul(x_tensor,weights),bias))


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    layer = conv2d_maxpool(x, conv_num_outputs=32, conv_ksize=(5, 5), conv_strides=(1, 1), pool_ksize=(2, 2),pool_strides=(2, 2))
    layer = conv2d_maxpool(layer, conv_num_outputs=64, conv_ksize=(5, 5), conv_strides=(1, 1), pool_ksize=(2, 2),pool_strides=(2, 2))
    #layer = conv2d_maxpool(layer, conv_num_outputs=128, conv_ksize=(3, 3), conv_strides=(1, 1), pool_ksize=(2, 2),pool_strides=(2, 2))

    # TODO: Apply a Flatten Layer
    layer = flatten(layer)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    layer = fully_conn(layer, 1024)
    layer = tf.nn.dropout(layer, keep_prob)
    # layer = fully_conn(layer, 256)
    # layer = tf.nn.dropout(layer, keep_prob)

    # TODO: Apply an Output Layer
    layer = fully_conn(layer, 10)
    return layer


def train_neural_network():
    epochs = 25
    batch_size = 128
    keep_probability = 0.5

    # Build the Neural Network
    tf.reset_default_graph()

    # Inputs
    x, y, keep_prob = get_input((32, 32, 3), 10)

    # Model
    logits = conv_net(x, keep_prob)
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    save_model_path = './image_classification'
    print('Training on whole dataset')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                    sess.run(optimizer, feed_dict={x:batch_features, y:batch_labels, keep_prob:keep_probability})
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                loss = sess.run(cost, feed_dict={x: batch_features, y: batch_labels, keep_prob: 1.})
                train_acc = sess.run(accuracy, feed_dict={x: batch_features , y: batch_labels, keep_prob: 1.})
                valid_acc = sess.run(accuracy, feed_dict={x: valid_features[:128], y: valid_labels[:128], keep_prob: 1.})
                print("Loss is {:<10.5f},  Train_acc is {:<10.5f},  Valid_acc is {:<10.5f}".format(loss, train_acc, valid_acc))
        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)


def test_model():
    save_model_path = './image_classification'
    batch_size = 128
    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0

        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels,batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total / test_batch_count))

if __name__ == '__main__':
    train_neural_network()