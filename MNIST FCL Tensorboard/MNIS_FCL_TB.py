import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def fcl(input,size_in,size_out,nom):
    with tf.name_scope(nom):
        W = tf.Variable(tf.truncated_normal([size_in,size_out], stddev=0.1), name="Weigths")
        b = tf.Variable(tf.constant(0.1,shape=[1,size_out]), name="Biases")
    with tf.name_scope("application-"+nom):    
        output = tf.matmul(input, W)
        output = tf.add(output, b)        
    weights_histo = tf.summary.histogram('weights', W)
    biases_histo = tf.summary.histogram('biases', b) 
    histo = {"weights": weights_histo, "biases": biases_histo}
    return output, histo

  


    
#Voir mnist hp opti
with tf.name_scope("Input"): 
    input_images = tf.placeholder(tf.float32, [None,784])

with tf.name_scope("Labels"): 
    labels = tf.placeholder(tf.float32, [None,10])


layer_1,histo1 = fcl(input_images,784,300,"couche-1")
layer_1 = tf.nn.relu(layer_1,name="relu-1") #max(x,0) 


layer_2 = fcl(layer_1,300,500,"couche-2")[0]
layer_2 = tf.nn.relu(layer_2,name="relu-2")


prediction = fcl(layer_2,500,10,"Couche-de-sortie")[0]

with tf.name_scope("Erreur"):
    erreur = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
with tf.name_scope("Train"):
    optimiseur = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(erreur)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        predictions_correctes = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(predictions_correctes, tf.float32))
    
erreur_scalaire = tf.summary.scalar('erreur',erreur)                   
accuracy_scalaire = tf.summary.scalar('accuracy',accuracy)


n_epochs = 10
batch_size = 1000

def tensorboard(epoch,flux_test,writer,sess):
    summ_biases = histo1["biases"].eval(flux_test)
    summ_weights = histo1["weights"].eval(flux_test)
    writer.add_summary(summ_biases, global_step=epoch)
    writer.add_summary(summ_weights, global_step=epoch)
    
      
    summ_acc=sess.run(accuracy_scalaire, feed_dict=flux_test)
    writer.add_summary(summ_acc, global_step=epoch)
    

    summ_erreur=sess.run(erreur_scalaire, feed_dict=flux_test)
    writer.add_summary(summ_erreur, global_step=epoch)
    return
    
    
def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        writer = tf.summary.FileWriter('./logs', graph=sess.graph) #----------       
        flux_test = {input_images:mnist.test.images, labels:mnist.test.labels}
        
        tensorboard(0,flux_test,writer,sess)        
        
        precision = sess.run(accuracy, feed_dict=flux_test)
        print("\nPrecision:", precision*100, "Epoch:", 0)
        
        
        for epoch in range(n_epochs):
            for i in range(int(len(mnist.train.images)/batch_size)):
                flux_image, flux_labels = mnist.train.next_batch(batch_size)
                flux_train = {input_images:flux_image, labels:flux_labels}
                sess.run(optimiseur, feed_dict=flux_train)
            
            
            tensorboard(epoch,flux_test,writer,sess)
            
            # summ_biases = histo1["biases"].eval(flux_test)
            # summ_weights = histo1["weights"].eval(flux_test)
            # writer.add_summary(summ_biases, global_step=epoch)
            # writer.add_summary(summ_weights, global_step=epoch)
            
              
            # summ_acc=sess.run(accuracy_scalaire, feed_dict=flux_test)
            # writer.add_summary(summ_acc, global_step=epoch)
            
       
            # summ_erreur=sess.run(erreur_scalaire, feed_dict=flux_test)
            # writer.add_summary(summ_erreur, global_step=epoch)
            
            precision = sess.run(accuracy, feed_dict=flux_test)
            print("\nPrecision:", precision*100, "Epoch:", epoch+1)
            
        writer.close() #------------

train()
