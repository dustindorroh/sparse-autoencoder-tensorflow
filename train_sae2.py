import tensorflow as tf
import matplotlib.pyplot
import math
import argparse

from SparseAutoEncoder import FeedforwardSparseAutoEncoder
from SparseAutoEncoder import visualizeW1
from SparseAutoEncoder import run_saved_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", help="Number of iterations to run", type=int,default=4000)
    parser.add_argument("--n-hidden", help="Size of hidden layer", type=int,default=256)
    parser.add_argument("--n-inputs", help="Size of inputs", type=int,default=1024)
    parser.add_argument("--input-model-dir", help="Directory of the first sae model", default='sae1')
    parser.add_argument("--export-dir", help="Directory to save the model to", default='output')
    parser.add_argument("--sparsity", help="sparsity parameter, typically a small value close to zero.",type=float, default=0.01)
    return parser.parse_args()

def main(n_iters,n_hidden,n_inputs,input_model_dir,export_dir,sparsity):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    reconstruction_result,hidden_result = run_saved_model(mnist.train.images,input_model_dir)

    start = 0
    lens = 1000

    sae = FeedforwardSparseAutoEncoder(n_inputs,n_hidden,rho=sparsity)
    sae.training(hidden_result[start:start+lens],n_iter=n_iters)

    # After training the model, an image of the representations (W1) will be saved
    # Please check trained4000.png for example
    images=sae.W1.eval(sae.sess)
    images=images.transpose()
    visualizeW1(images,n_inputs**.5,10,n_iters,file_name=export_dir)

    tf.saved_model.simple_save(sae.sess
                              ,export_dir
                              ,inputs={'inputs': sae.inputs}
                              ,outputs={'hidden': sae.hidden,'reconstruction': sae.outputs})


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(**vars(args))

