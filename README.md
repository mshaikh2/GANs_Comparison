# MNIST reconstruction using Convnet, Neuralnet and CapsuleNets


## Deep Convolutionalnet GAN
The below GIF displays the sample of images generated from epoch 1 to 50 at every 5 epochs.

Conv layers enable GANs to generate better images much faster than neural net.

Each epoch takes around 60 seconds

![Images_generated_using_conv_net](/images/gan_cnn/digits/cnn_epoch_1_50.gif?raw=true "Images Generated using Conv Layers in GAN architecture")

### Graph of Loss over 50 epochs
![Graph1](/images/gan_cnn/conv_gan_loss.png?raw=true "Graph of the loss over 50 epochs")

## Deep Neuralnet GAN
The below GIF displays the sample of images generated from epoch 1 to 200 at every 20 epochs.

Neural net enables GANs to generate decent images but after much longer training epochs.

Each epoch takes around 15 seconds.

![Images_generated_using_conv_net](/images/gan_neuralnet/digits/gan_nn_epoch_1_to_200.gif?raw=true "Images Generated using NeuralNet Layers in GAN architecture")

## Capsule Nets
The below GIF displays the sample of images generated from epoch 1 to 9 at every epoch.

At the decoder end a 28x28 image is reconstructed by passing the latent vector along with its true class variable through two fully connected layers

Each epoch takes around 55 mins seconds.

![Images_generated_using_caps_net](/images/capsulenet/Selected/epochs.gif?raw=true "Images Generated using CapsNet")

### Graph of Loss over 9 epochs
![Graph3](/images/capsulenet/capsnet_graph.jpg?raw=true "Graph of the loss and accuracy over 9 epochs")

## Libraries
#### Tensorflow
#### Keras
#### openCV
#### PIL
#### numpy

## Refrences
#### [1] GANs, https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
#### [2] https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners
#### [3] https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f
#### [4] https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/
#### [5] https://kndrck.co/posts/capsule_networks_explained/
#### [6] https://ctmakro.github.io/site/on_learning/fast_gan_in_keras.html
#### [7] Overview of GANs, https://arxiv.org/pdf/1710.07035.pdf
#### [8] Capsule Nets, https://arxiv.org/pdf/1710.09829.pdf
#### [9] https://github.com/XifengGuo/CapsNet-Keras


