import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def fcl(input,size_in,size_out):
    W = tf.Variable(tf.truncated_normal([size_in,size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1,shape=[1,size_out]))
    output = tf.matmul(input, W)
    output = tf.add(output, b)
    return output

with tf.device('/CPU:0'):   
    input_images = tf.placeholder(tf.float32, [None,784])
    labels = tf.placeholder(tf.float32, [None,10])

    layer_1 = fcl(input_images,784,300)
    layer_1 = tf.nn.relu(layer_1) #max(x,0) 

    layer_2 = fcl(layer_1,300,500)
    layer_2 = tf.nn.relu(layer_2)

    prediction = fcl(layer_2,500,10)

    erreur = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
    optimiseur = tf.train.AdamOptimizer().minimize(erreur)

    predictions_correctes = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(predictions_correctes, tf.float32))

n_epochs = 100
batch_size = 2048

# config = tf.ConfigProto(device_count = {"CPU":
def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        flux_test = {input_images:mnist.test.images, labels:mnist.test.labels}
        precision = sess.run(accuracy, feed_dict=flux_test)
        print("\nPrecision:", precision*100, "Epoch:", 0)
        for epoch in range(n_epochs):
            for i in range(int(len(mnist.train.images)/batch_size)):
                flux_image, flux_labels = mnist.train.next_batch(batch_size)
                flux_train = {input_images:flux_image, labels:flux_labels}
                sess.run(optimiseur, feed_dict=flux_train)
                
            precision = sess.run(accuracy, feed_dict=flux_test)
            print("\nPrecision:", precision*100, "Epoch:", epoch+1)
        

train()
