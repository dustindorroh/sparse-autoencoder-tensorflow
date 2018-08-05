# Deep Neural Network with sparse auto encoders

Attempt at creating a DNN with a similar architecture found here:
https://arxiv.org/abs/1605.00129

Three SAEs are used to formulate the hidden layers of the DNN and then a 
logistic regression layer.

Three SAEs are trained with hidden layers being inputs to the next SAE.
 ![](3SAEs.png)

## Sparse Autoencoder with Tensorflow
The implentationof the SAE follows the description found here: 
https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf

## Example training
```bash
python train_sae1.py --n-iters 4000 --n-hidden 1024 --export-dir sae1 --sparsity 0.15
python train_sae2.py --n-iters 4000 --n-hidden 256 --n-inputs 1024 --input-model-dir sae1 --export-dir sae2 --sparsity 0.15
python train_sae3.py --n-iters 4000 --n-hidden 64  --n-inputs 256  --input-model-dirs sae1 sae2 --export-dir sae3 --sparsity 0.1
```
