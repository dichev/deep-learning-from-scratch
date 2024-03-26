## Deep learning from scratch

> "What I cannot create, I do not understand." -- Richard Feynman

I agree.


---
 Everything is coded from scratch, except:
* using PyTorch's tensors for GPU computation
* using PyTorch's autograd for the backpropagation


<!-- auto-generated-start -->


### Layers

`lib.layers` [➜](src/lib/layers.py)
- Linear
- Embedding
- BatchNorm <sup>[*[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift]*](https://proceedings.mlr.press/v37/ioffe15.pdf)</sup>
- BatchNorm1d
- BatchNorm2d
- LayerNorm <sup>[*[Layer Normalization]*](https://arxiv.org/pdf/1607.06450.pdf)</sup>
- LocalResponseNorm <sup>[*[ImageNet Classification with Deep Convolutional Neural Networks]*](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)</sup>
- Dropout <sup>[*[Dropout: A Simple Way to Prevent Neural Networks from Overfitting]*](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)</sup>
- RNN_cell
- LSTM_cell <sup>[*[Generating Sequences With Recurrent Neural Networks]*](https://arxiv.org/pdf/1308.0850.pdf)</sup>
- GRU_cell
- RNN
- Conv2d
- Conv2dGroups
- Pool2d
- MaxPool2d
- AvgPool2d
- BatchAddPool
- SEGate <sup>[*[Squeeze-and-Excitation Gate layer]*](https://arxiv.org/pdf/1709.01507.pdf)</sup>
- Graph_cell
- GCN_cell <sup>[*[Semi-Supervised Classification with Graph Convolutional Networks]*](https://arxiv.org/pdf/1609.02907.pdf)</sup>
- GraphSAGE_cell <sup>[*[Inductive Representation Learning on Large Graphs]*](https://arxiv.org/pdf/1706.02216.pdf)</sup>
- DiffPool <sup>[*[Hierarchical Graph Representation Learning with Differentiable Pooling]*](https://proceedings.neurips.cc/paper_files/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf)</sup>
- ReLU
- Flatten

`lib.autoencoders` [➜](src/lib/autoencoders.py)
- MatrixFactorization
- AutoencoderLinear
- Word2Vec <sup>[*[Efficient Estimation of Word Representations in Vector Space]*](https://arxiv.org/pdf/1301.3781.pdf)</sup>


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
- LSTM <sup>[*[Generating Sequences With Recurrent Neural Networks]*](https://arxiv.org/pdf/1308.0850.pdf)</sup>
- GRU
- LangModel
- EchoStateNetwork
- Encoder
- Decoder
- Seq2Seq <sup>[*[Sequence to Sequence Learning with Neural Networks]*](https://papers.nips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)</sup>

`models.convolutional_networks` [➜](src/models/convolutional_networks.py)
- SimpleCNN
- SimpleFullyCNN
- LeNet5 <sup>[*[Gradient-based learning applied to document recognition]*](https://hal.science/hal-03926082/document)</sup>
- AlexNet <sup>[*[ImageNet Classification with Deep Convolutional Neural Networks]*](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)</sup>
- NetworkInNetwork <sup>[*[Network In Network]*](https://arxiv.org/pdf/1312.4400.pdf)</sup>
- VGG16 <sup>[*[Very Deep Convolutional Networks for Large-Scale Image Recognition]*](https://arxiv.org/pdf/1409.1556.pdf)</sup>
- GoogLeNet <sup>[*[Going deeper with convolutions]*](https://arxiv.org/pdf/1409.4842.pdf?)</sup>
- DeepPlainCNN

`models.residual_networks` [➜](src/models/residual_networks.py)
- ResNet34 <sup>[*[Deep Residual Learning for Image Recognition]*](https://arxiv.org/pdf/1512.03385.pdf)</sup>
- ResNet50 <sup>[*[Deep Residual Learning for Image Recognition]*](https://arxiv.org/pdf/1512.03385.pdf)</sup>
- ResNeXt50 <sup>[*[Aggregated Residual Transformations for Deep Neural Networks]*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf)</sup>
- SEResNet50 <sup>[*[Squeeze-and-Excitation Networks]*](https://arxiv.org/pdf/1709.01507.pdf)</sup>
- SEResNeXt50 <sup>[*[Squeeze-and-Excitation Networks]*](https://arxiv.org/pdf/1709.01507.pdf)</sup>
- DenseNet121 <sup>[*[Densely Connected Convolutional Networks]*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)</sup>

`models.graph_networks` [➜](src/models/graph_networks.py)
- GCN <sup>[*[Semi-Supervised Classification with Graph Convolutional Networks]*](https://arxiv.org/pdf/1609.02907.pdf)</sup>
- GraphSAGE <sup>[*[Inductive Representation Learning on Large Graphs]*](https://arxiv.org/pdf/1706.02216.pdf)</sup>
- GIN <sup>[*[How Powerful are Graph Neural Networks?]*](https://arxiv.org/pdf/1810.00826v3.pdf)</sup>
- DiffPoolNet <sup>[*[Hierarchical Graph Representation Learning with Differentiable Pooling]*](https://proceedings.neurips.cc/paper_files/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf)</sup>

`models.attention_networks` [➜](src/models/attention_networks.py)
- RecurrentAttention <sup>[*[Recurrent Models of Visual Attention]*](https://arxiv.org/pdf/1406.6247.pdf)</sup>
- SpatialTransformer <sup>[*[Spatial Transformer Networks]*](https://arxiv.org/pdf/1506.02025.pdf)</sup>
- SpatialTransformerNet

`models.blocks.convolutional_blocks` [➜](src/models/blocks/convolutional_blocks.py)
- Inception <sup>[*[Going deeper with convolutions]*](https://arxiv.org/pdf/1409.4842.pdf?)</sup>
- ResBlock <sup>[*[Deep Residual Learning for Image Recognition]*](https://arxiv.org/pdf/1512.03385.pdf)</sup>
- ResBottleneckBlock <sup>[*[Deep Residual Learning for Image Recognition]*](https://arxiv.org/pdf/1512.03385.pdf)</sup>
- ResNeXtBlock <sup>[*[Aggregated Residual Transformations for Deep Neural Networks]*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf)</sup>
- DenseLayer <sup>[*[Densely Connected Convolutional Networks]*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)</sup>
- DenseBlock <sup>[*[Densely Connected Convolutional Networks]*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)</sup>
- DenseTransition <sup>[*[Densely Connected Convolutional Networks]*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)</sup>


### Example usages
- examples/ [➜](examples/)
- examples/convolutional [➜](examples/convolutional)
- examples/energy_based [➜](examples/energy_based)
- examples/graph [➜](examples/graph)
- examples/recurrent [➜](examples/recurrent)
- examples/attention [➜](examples/attention)
- examples/shallow [➜](examples/shallow)

<!-- auto-generated-end -->

## Installation
```
conda env create --name dev --file=./environment.yml
```
