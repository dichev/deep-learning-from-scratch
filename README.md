## Deep learning from scratch

> "What I cannot create, I do not understand." -- Richard Feynman

I agree.


---
 Everything is coded from scratch, except:
* using PyTorch's tensors for GPU computation
* using PyTorch's autograd for the backpropagation
 


<style>
    i {
        color: rgba(128, 128, 128, 0.4);
        font-style: normal;
    }
</style>


### Layers
- <i>lib.layers.</i>Linear
- <i>lib.layers.</i>Embedding
- <i>lib.layers.</i>BatchNorm1d
- <i>lib.layers.</i>BatchNorm2d
- <i>lib.layers.</i>LayerNorm
- <i>lib.layers.</i>LocalResponseNorm
- <i>lib.layers.</i>Dropout
- <i>lib.layers.</i>RNN_cell
- <i>lib.layers.</i>LSTM_cell
- <i>lib.layers.</i>GRU_cell
- <i>lib.layers.</i>RNN
- <i>lib.layers.</i>Conv2d
- <i>lib.layers.</i>Conv2dGroups
- <i>lib.layers.</i>Pool2d
- <i>lib.layers.</i>MaxPool2d
- <i>lib.layers.</i>AvgPool2d
- <i>lib.layers.</i>SEGate ([Squeeze-and-Excitation Gate layer](https://arxiv.org/pdf/1709.01507.pdf))
- <i>lib.layers.</i>ModuleList
- <i>lib.layers.</i>Sequential
- <i>lib.layers.</i>ReLU
- <i>lib.layers.</i>Flatten
- <i>lib.autoencoders.</i>MatrixFactorization
- <i>lib.autoencoders.</i>AutoencoderLinear
- <i>lib.autoencoders.</i>Word2Vec

### Optimizers
- <i>lib.optimizers.</i>SGD
- <i>lib.optimizers.</i>SGD_Momentum
- <i>lib.optimizers.</i>AdaGrad
- <i>lib.optimizers.</i>RMSProp
- <i>lib.optimizers.</i>AdaDelta
- <i>lib.optimizers.</i>Adam
- <i>lib.optimizers.</i>LR_Scheduler
- <i>lib.optimizers.</i>LR_StepScheduler
- <i>lib.optimizers.</i>LR_PlateauScheduler

### Regularizers
- <i>lib.regularizers.</i>L2_regularizer
- <i>lib.regularizers.</i>L1_regularizer
- <i>lib.regularizers.</i>elastic_regularizer
- <i>lib.regularizers.</i>max_norm_constraint_
- <i>lib.regularizers.</i>grad_clip_
- <i>lib.regularizers.</i>grad_clip_norm_

### Models / Networks
**Shallow models**:
- <i>models.shallow_models.</i>Perceptron
- <i>models.shallow_models.</i>SVM
- <i>models.shallow_models.</i>LeastSquareRegression
- <i>models.shallow_models.</i>LogisticRegression
- <i>models.shallow_models.</i>MulticlassPerceptron
- <i>models.shallow_models.</i>MulticlassSVM
- <i>models.shallow_models.</i>MultinomialLogisticRegression

**Energy based models**:
- <i>models.energy_based_models.</i>HopfieldNetwork
- <i>models.energy_based_models.</i>HopfieldNetworkOptimized
- <i>models.energy_based_models.</i>RestrictedBoltzmannMachine

**Recurrent networks**:
- <i>models.recurrent_networks.</i>SimpleRNN
- <i>models.recurrent_networks.</i>LSTM
- <i>models.recurrent_networks.</i>GRU
- <i>models.recurrent_networks.</i>EchoStateNetwork


**Convolutional networks:** 

- <i>models.convolutional_networks.</i>SimpleCNN
- <i>models.convolutional_networks.</i>SimpleFullyCNN
- <i>models.convolutional_networks.</i>LeNet5 ([Gradient-based learning applied to document recognition](https://hal.science/hal-03926082/document))
- <i>models.convolutional_networks.</i>AlexNet ([ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf))
- <i>models.convolutional_networks.</i>NetworkInNetwork ([Network In Network](https://arxiv.org/pdf/1312.4400.pdf))
- <i>models.convolutional_networks.</i>VGG16 ([Very Deep Convolutional Networks fpr Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf))
- <i>models.convolutional_networks.</i>GoogLeNet ([Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf?))
- <i>models.convolutional_networks.</i>DeepPlainCNN
- <i>models.blocks.convolutional_blocks.</i>Inception

**Residual networks:**
- <i>models.residual_networks.</i>ResBlock
- <i>models.residual_networks.</i>ResBottleneckBlock
- <i>models.residual_networks.</i>ResNeXtBlock
- <i>models.residual_networks.</i>ResNet34 ([Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf))
- <i>models.residual_networks.</i>ResNet50 ([Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf))
- <i>models.residual_networks.</i>ResNeXt50 ([Aggregated Residual Transformations for Deep Neural Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf))
- <i>models.residual_networks.</i>SEResNet50 ([Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf))
- <i>models.residual_networks.</i>SEResNeXt50 ([Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf))
- <i>models.residual_networks.</i>DenseNet121 ([Densely Connected Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf))
- <i>models.blocks.convolutional_blocks.</i>ResBlock
- <i>models.blocks.convolutional_blocks.</i>ResBottleneckBlock
- <i>models.blocks.convolutional_blocks.</i>ResNeXtBlock
- <i>models.blocks.convolutional_blocks.</i>DenseLayer
- <i>models.blocks.convolutional_blocks.</i>DenseBlock
- <i>models.blocks.convolutional_blocks.</i>DenseTransition

## Example usages
- <i>examples/convolutional/</i>cifar10_classification.py
- <i>examples/convolutional/</i>conv_filters_preview.py
- <i>examples/energy_based/</i>hopfield_network_memorize_letters.py
- <i>examples/energy_based/</i>hopfield_network_optimized_memorize_images.py
- <i>examples/energy_based/</i>restricted_boltzmann_memorize_images.py
- <i>examples/recurrent/</i>rnn_masked_words.py
- <i>examples/recurrent/</i>rnn_predict_next_char.py
- <i>examples/shallow/</i>binary_classification.py
- <i>examples/shallow/</i>matrix_factorization.py
- <i>examples/shallow/</i>mnist_classification.py
- <i>examples/shallow/</i>multiclass_classification.py
- <i>examples/</i>visualize_optimizers.py
- <i>examples/</i>word2vec.py
