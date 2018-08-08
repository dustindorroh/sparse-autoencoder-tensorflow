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
    parser.add_argument("--n-hidden", help="Size of hidden layer", type=int,default=64)
    parser.add_argument("--n-inputs", help="Size of inputs", type=int, default=256)
    parser.add_argument("--input-model-dirs", help="Directores to the first and second sae model", metavar='dir',nargs=2, default=('sae1','sae2'))
    parser.add_argument("--export-dir", help="Directory to save the model to", default='output')
    parser.add_argument("--sparsity", help="sparsity parameter, typically a small value close to zero.",type=float, default=0.01)
    return parser.parse_args()

def main(n_iters,n_hidden,n_inputs,input_model_dirs,export_dir,sparsity):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    input_model_dir_sae1,input_model_dir_sae2 = input_model_dirs
    reconstruction_result,hidden_result = run_saved_model(mnist.train.images,input_model_dir_sae1)
    print 'Sae1'
    print 'reconstruction_result.shape {}\nhidden_result.shape {}'.format(reconstruction_result.shape,hidden_result.shape)
    reconstruction_result,hidden_result = run_saved_model(hidden_result,input_model_dir_sae2)
    print 'Sae2'
    print 'reconstruction_result.shape {}\nhidden_result.shape {}'.format(reconstruction_result.shape,hidden_result.shape)

    start = 0
    lens = 1000

    sae = FeedforwardSparseAutoEncoder(n_inputs,n_hidden,rho=sparsity)
    sae.training(hidden_result[start:start+lens],n_iter=n_iters)

    # After training the model, an image of the representations (W1) will be saved
    # Please check trained4000.png for example
    images = sae.W1.eval(sae.sess)
    print 'W1.shape {}, min {}, max {}'.format(images.shape,images.min(),images.max())
    #images = images.transpose()
    print 'images.shape {}'.format(images.shape)
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

