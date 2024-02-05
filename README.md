## Deep learning from scratch

> "What I cannot create, I do not understand." -- Richard Feynman

I agree.


---
 Everything is coded from scratch, except:
* using PyTorch's tensors for GPU computation
* using PyTorch's autograd for the backpropagation


<!-- auto-generated-bellow -->

### Layers

`lib.layers` [➜](src/lib/layers.py)
- Linear
- Embedding
- BatchNorm1d ([*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*](https://proceedings.mlr.press/v37/ioffe15.pdf))
- BatchNorm2d ([*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*](https://proceedings.mlr.press/v37/ioffe15.pdf))
- LayerNorm ([*Layer Normalization*](https://arxiv.org/pdf/1607.06450.pdf))
- LocalResponseNorm ([*ImageNet Classification with Deep Convolutional Neural Networks*](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf))
- Dropout ([*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf))
- RNN_cell
- LSTM_cell ([*Generating Sequences With Recurrent Neural Networks*](https://arxiv.org/pdf/1308.0850.pdf))
- GRU_cell
- RNN
- Conv2d
- Conv2dGroups
- Pool2d
- MaxPool2d
- AvgPool2d
- BatchAddPool
- SEGate ([*Squeeze-and-Excitation Gate layer*](https://arxiv.org/pdf/1709.01507.pdf))
- GraphLayer
- GraphAddLayer
- GraphConvLayer ([*Semi-Supervised Classification with Graph Convolutional Networks*](https://arxiv.org/pdf/1609.02907.pdf))
- GraphSAGELayer ([*Inductive Representation Learning on Large Graphs*](https://arxiv.org/pdf/1706.02216.pdf))
- ReLU
- Flatten

`lib.autoencoders` [➜](src/lib/autoencoders.py)
- MatrixFactorization
- AutoencoderLinear
- Word2Vec ([*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/pdf/1301.3781.pdf))


### Optimizers

`lib.optimizers` [➜](src/lib/optimizers.py)
- Optimizer
- SGD
- SGD_Momentum
- AdaGrad
- RMSProp
- AdaDelta
- Adam
- LR_Scheduler
- LR_StepScheduler
- LR_PlateauScheduler


### Models / Networks

`models.shallow_models` [➜](src/models/shallow_models.py)
- Perceptron
- SVM
- LeastSquareRegression
- LogisticRegression
- MulticlassPerceptron
- MulticlassSVM
- MultinomialLogisticRegression

`models.energy_based_models` [➜](src/models/energy_based_models.py)
- HopfieldNetwork
- HopfieldNetworkOptimized
- RestrictedBoltzmannMachine

`models.recurrent_networks` [➜](src/models/recurrent_networks.py)
- RNN_factory
- SimpleRNN
- LSTM ([*Generating Sequences With Recurrent Neural Networks*](https://arxiv.org/pdf/1308.0850.pdf))
- GRU
- EchoStateNetwork

`models.convolutional_networks` [➜](src/models/convolutional_networks.py)
- SimpleCNN
- SimpleFullyCNN
- LeNet5 ([*Gradient-based learning applied to document recognition*](https://hal.science/hal-03926082/document))
- AlexNet ([*ImageNet Classification with Deep Convolutional Neural Networks*](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf))
- NetworkInNetwork ([*Network In Network*](https://arxiv.org/pdf/1312.4400.pdf))
- VGG16 ([*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/pdf/1409.1556.pdf))
- GoogLeNet ([*Going deeper with convolutions*](https://arxiv.org/pdf/1409.4842.pdf?))
- DeepPlainCNN

`models.residual_networks` [➜](src/models/residual_networks.py)
- ResNet34 ([*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385.pdf))
- ResNet50 ([*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385.pdf))
- ResNeXt50 ([*Aggregated Residual Transformations for Deep Neural Networks*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf))
- SEResNet50 ([*Squeeze-and-Excitation Networks*](https://arxiv.org/pdf/1709.01507.pdf))
- SEResNeXt50 ([*Squeeze-and-Excitation Networks*](https://arxiv.org/pdf/1709.01507.pdf))
- DenseNet121 ([*Densely Connected Convolutional Networks*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf))

`models.graph_networks` [➜](src/models/graph_networks.py)
- GIN ([*How Powerful are Graph Neural Networks?*](https://arxiv.org/pdf/1810.00826v3.pdf))

`models.blocks.convolutional_blocks` [➜](src/models/blocks/convolutional_blocks.py)
- Inception ([*Going deeper with convolutions*](https://arxiv.org/pdf/1409.4842.pdf?))
- ResBlock ([*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385.pdf))
- ResBottleneckBlock ([*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385.pdf))
- ResNeXtBlock ([*Aggregated Residual Transformations for Deep Neural Networks*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf))
- DenseLayer ([*Densely Connected Convolutional Networks*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf))
- DenseBlock ([*Densely Connected Convolutional Networks*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf))
- DenseTransition ([*Densely Connected Convolutional Networks*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf))


### Example usages
- examples/ [➜](examples/)
- examples/convolutional [➜](examples/convolutional)
- examples/energy_based [➜](examples/energy_based)
- examples/graph [➜](examples/graph)
- examples/recurrent [➜](examples/recurrent)
- examples/shallow [➜](examples/shallow)
