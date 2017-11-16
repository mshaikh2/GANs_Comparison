# BASIC Comparision GAN for MNIST using Convnet and Neuralnet


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

### Graph of Loss over 200 epochs
![Graph2](/images/gan_neuralnet/gan_neural_loss.png?raw=true "Graph of the loss over 200 epochs")

## Libraries
Tensorflow

Keras

openCV

PIL

numpy




