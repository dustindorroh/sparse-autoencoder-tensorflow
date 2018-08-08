import tensorflow as tf
import matplotlib.pyplot
import math
import argparse

from SparseAutoEncoder import FeedforwardSparseAutoEncoder
from SparseAutoEncoder import visualizeW1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", help="Number of iterations to run", type=int,default=4000)
    parser.add_argument("--n-hidden", help="Dimension of hidden layer", type=int,default=100)
    parser.add_argument("--export-dir", help="Directory to save the model to", default='output')
    parser.add_argument("--sparsity", help="sparsity parameter, typically a small value close to zero.",type=float, default=0.01)

    return parser.parse_args()

def main(n_iters,n_hidden,export_dir,sparsity):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    n_inputs = 784
    start = 0
    lens = 1000

    sae = FeedforwardSparseAutoEncoder(n_inputs,n_hidden,rho=sparsity)
    sae.training(mnist.train.images[start:start+lens],n_iter=n_iters)

    # After training the model, an image of the representations (W1) will be saved
    # Please check trained4000.png for example
    images=sae.W1.eval(sae.sess)
    print 'W1.shape {}, min {}, max {}'.format(images.shape,images.min(),images.max())
    images=images.transpose()
    
    try:
        visualizeW1(images,int(n_inputs**.5),5,n_iters,file_name=export_dir+'_')
    except ValueError:
        visualizeW1(images,int(n_hidden**.5),5,n_iters,file_name=export_dir+'_')
        
    tf.saved_model.simple_save(sae.sess
                              ,export_dir
                              ,inputs={'inputs': sae.inputs}
                              ,outputs={'hidden': sae.hidden,'reconstruction': sae.outputs})


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(**vars(args))

