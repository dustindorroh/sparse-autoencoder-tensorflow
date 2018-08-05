# This is an example of using Tensorflow to build Sparse Autoencoder
# for representation learning.
# It is the implementation of the sparse autoencoder for
#        https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
#
# For any enquiry, please contact Dr. Zhiwei Lin  at Ulster University
#       http://scm.ulster.ac.uk/zhiwei.lin/
#
#
# ==============================================================================
import tensorflow as tf
import matplotlib.pyplot
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--n-iters", help="Number of iterations to run", type=int,default=4000)
    return parser.parse_args()

def kl_divergence(rho, rho_hat):
    return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)


class FeedforwardSparseAutoEncoder(object):
    '''
      This is the implementation of the sparse autoencoder for https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf

    '''
    def __init__(self, n_input, n_hidden, rho=0.01, alpha=0.0001, beta=3, activation=tf.nn.sigmoid, learning_rate=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.rho = rho  # sparse parameters
        self.alpha = alpha
        self.beta = beta
        self.activation = activation

        # Setup weight initializer
        self.init_weights = tf.contrib.layers.xavier_initializer()
        
        self.inputs = tf.placeholder('float',shape=[None,self.n_input])

        self.hidden = self.encode(self.inputs)
        
        self.outputs = self.decode(self.hidden)
        
        self.loss = self.loss_func()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.sess = tf.Session()

    def encode(self,X):
        # Create weights and bias
        self.W1 = tf.Variable(self.init_weights((self.n_input,self.n_hidden)))
        self.b1 = tf.Variable(self.init_weights((1,self.n_hidden)))
        return self.activation(tf.matmul(X, self.W1) + self.b1)

    def decode(self,H):
        self.W2 = self.init_weights((self.n_hidden,self.n_input))
        self.b2 = self.init_weights((1,self.n_input))
        return self.activation(tf.matmul(H, self.W2) + self.b2)

    def kl_divergence(self, rho, rho_hat):
        return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

    def loss_func(self):
        # Build cost function
        # Average hidden layer over all data points in X, 
        # Page 14 in https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
        rho_hat = tf.reduce_mean(self.hidden,axis=0)
        kl = self.kl_divergence(self.rho, rho_hat)

        # cost
        #cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.inputs - self.outputs), 2.0)) + self.beta*tf.reduce_sum(kl)
        cost = 0.5*tf.reduce_mean(tf.reduce_sum((self.inputs - self.outputs)**2,axis=1))  \
              +0.5*self.alpha*(tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2))   \
              +self.beta*tf.reduce_sum(kl)
        return cost

    def training(self,training_data,  n_iter=100):
        
        opt = self.optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for i in xrange(n_iter):
            cost,_= self.sess.run((self.loss,opt),feed_dict={ self.inputs: training_data})
            if i % 10 == 0:
                print('Iter {}/{}  loss: {}'.format(i,n_iter,cost))

def visualizeW1(images, vis_patch_side, hid_patch_side, iter, file_name="trained_"):
    """ Visual all images in one pane"""

    figure, axes = matplotlib.pyplot.subplots(nrows=hid_patch_side, ncols=hid_patch_side)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """

        image = axis.imshow(images[index, :].reshape(vis_patch_side, vis_patch_side),
                            cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """
    file=file_name+str(iter)+".png"
    matplotlib.pyplot.savefig(file)
    print("Written into "+ file)
    matplotlib.pyplot.close()


def main(n_iters):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    n_inputs = 784
    n_hidden = 100
    start = 0
    lens = 1000

    sae = FeedforwardSparseAutoEncoder(n_inputs,n_hidden)
    sae.training(mnist.train.images[start:start+lens],n_iter=n_iters)

    # After training the model, an image of the representations (W1) will be saved
    # Please check trained4000.png for example
    images=sae.W1.eval(sae.sess)
    images=images.transpose()
    visualizeW1(images,28,10,n_iters)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(**vars(args))

