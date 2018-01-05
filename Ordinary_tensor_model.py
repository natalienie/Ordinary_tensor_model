import tensorflow as tf
'''
import RNN_feature_label_prepare.creat_features_labels as cfl
import RNN_feature_label_prepare.get_coin_data as gcd

be alert that before running the train_neural_network function,
the RNN_feature_label_prepart.creat_features_labels must by run first
X_train, y_train, X_test, y_test must be assiged before training the model

'''

n_hl1 = 100
n_hl2 = 90
n_hl3 = 70
n_classes = 3
batch_size = 128
x = tf.placeholder('float', [None, 256])
y = tf.placeholder('float')

''''
i will first try to unpack the
128 * 2 feature into 256 array this procedured shall be done in the next_batch function below
a next_batch function serves to fetch the next_batch of features and labels

''''
def next_batch_X(data, i, batch_size):
    data = data.reshape(-1, 256)
    return np.array(data[i*batch_size:(i+1)*batch_size])

def next_batch_y(data, i, batch_size):
    return np.array(data[i*batch_size:(i+1)*batch_size])

def neural_network_model(data):

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([256, n_hl1])),
                      'bias': tf.Variable(tf.random_normal([n_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_hl1, n_hl2])),
                      'bias': tf.Variable(tf.random_normal([n_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_hl2, n_hl3])),
                      'bias': tf.Variable(tf.random_normal([n_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_hl3, n_classes])),
                      'bias': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(data, hidden_2_layer['weights']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(data, hidden_3_layer['weights']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)
    output = tf.matmul(l3, output_layer['weights']) + output_layer['bias']

def train_neural_network(x):

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 3
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            hm_batch = int(len(x)/batch_size)
            for i in range(hm_batch):
                epoch_x = next_batch_X(X_train, i, batch_size)
                epoch_y = next_batch_y(y_train, i, batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoth_loss += c
                i += 1
            print('Epoth', epoth, 'completed out of', hm_epochs, 'loss:', epoth_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy': , accuracy.eval({x: X_test, y: y_test}))
