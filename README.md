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
- BatchNorm <sup>[*[1]*](#ref1 "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift")</sup>
- BatchNorm1d
- BatchNorm2d
- LayerNorm <sup>[*[2]*](#ref2 "Layer Normalization")</sup>
- LocalResponseNorm <sup>[*[3]*](#ref3 "ImageNet Classification with Deep Convolutional Neural Networks")</sup>
- Dropout <sup>[*[4]*](#ref4 "Dropout: A Simple Way to Prevent Neural Networks from Overfitting")</sup>
- RNN_cell
- LSTM_cell <sup>[*[5]*](#ref5 "Generating Sequences With Recurrent Neural Networks")</sup>
- GRU_cell
- RNN
- Conv2d
- Conv2dGroups
- Pool2d
- MaxPool2d
- AvgPool2d
- BatchAddPool
- SEGate <sup>[*[6]*](#ref6 "Squeeze-and-Excitation Gate layer")</sup>
- Graph_cell
- GCN_cell <sup>[*[7]*](#ref7 "Semi-Supervised Classification with Graph Convolutional Networks")</sup>
- GraphSAGE_cell <sup>[*[8]*](#ref8 "Inductive Representation Learning on Large Graphs")</sup>
- DiffPool <sup>[*[9]*](#ref9 "Hierarchical Graph Representation Learning with Differentiable Pooling")</sup>
- ReLU
- GELU <sup>[*[10]*](#ref10 "Gaussian Error Linear Units (GELUs)")</sup>
- Flatten
- DotProductAttention
- AdditiveAttention <sup>[*[11]*](#ref11 "Neural Machine Translation by Jointly Learning to Align and Translate")</sup>
- MultiHeadAttention <sup>[*[12]*](#ref12 "Attention Is All You Need")</sup>
- PositionalEncoding <sup>[*[12]*](#ref12 "Attention Is All You Need")</sup>

`lib.autoencoders` [➜](src/lib/autoencoders.py)
- MatrixFactorization
- AutoencoderLinear
- Word2Vec <sup>[*[13]*](#ref13 "Efficient Estimation of Word Representations in Vector Space")</sup>


### Optimizers

`lib.optimizers` [➜](src/lib/optimizers.py)
- Optimizer
- SGD
- SGD_Momentum
- AdaGrad
- RMSProp
- AdaDelta
- Adam
- AdamW
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
- LSTM <sup>[*[5]*](#ref5 "Generating Sequences With Recurrent Neural Networks")</sup>
- GRU
- LangModel
- EchoStateNetwork
- Encoder
- Decoder
- Seq2Seq <sup>[*[14]*](#ref14 "Sequence to Sequence Learning with Neural Networks")</sup>

`models.convolutional_networks` [➜](src/models/convolutional_networks.py)
- SimpleCNN
- SimpleFullyCNN
- LeNet5 <sup>[*[15]*](#ref15 "Gradient-based learning applied to document recognition")</sup>
- AlexNet <sup>[*[3]*](#ref3 "ImageNet Classification with Deep Convolutional Neural Networks")</sup>
- NetworkInNetwork <sup>[*[16]*](#ref16 "Network In Network")</sup>
- VGG16 <sup>[*[17]*](#ref17 "Very Deep Convolutional Networks for Large-Scale Image Recognition")</sup>
- GoogLeNet <sup>[*[18]*](#ref18 "Going deeper with convolutions")</sup>
- DeepPlainCNN

`models.residual_networks` [➜](src/models/residual_networks.py)
- ResNet34 <sup>[*[19]*](#ref19 "Deep Residual Learning for Image Recognition")</sup>
- ResNet50 <sup>[*[19]*](#ref19 "Deep Residual Learning for Image Recognition")</sup>
- ResNeXt50 <sup>[*[20]*](#ref20 "Aggregated Residual Transformations for Deep Neural Networks")</sup>
- SEResNet50 <sup>[*[21]*](#ref21 "Squeeze-and-Excitation Networks")</sup>
- SEResNeXt50 <sup>[*[21]*](#ref21 "Squeeze-and-Excitation Networks")</sup>
- DenseNet121 <sup>[*[22]*](#ref22 "Densely Connected Convolutional Networks")</sup>

`models.graph_networks` [➜](src/models/graph_networks.py)
- GCN <sup>[*[7]*](#ref7 "Semi-Supervised Classification with Graph Convolutional Networks")</sup>
- GraphSAGE <sup>[*[8]*](#ref8 "Inductive Representation Learning on Large Graphs")</sup>
- GIN <sup>[*[23]*](#ref23 "How Powerful are Graph Neural Networks?")</sup>
- DiffPoolNet <sup>[*[9]*](#ref9 "Hierarchical Graph Representation Learning with Differentiable Pooling")</sup>

`models.attention_networks` [➜](src/models/attention_networks.py)
- RecurrentAttention <sup>[*[24]*](#ref24 "Recurrent Models of Visual Attention")</sup>
- SpatialTransformer <sup>[*[25]*](#ref25 "Spatial Transformer Networks")</sup>
- SpatialTransformerNet
- AttentionEncoder <sup>[*[11]*](#ref11 "Neural Machine Translation by Jointly Learning to Align and Translate")</sup>
- AttentionDecoder <sup>[*[11]*](#ref11 "Neural Machine Translation by Jointly Learning to Align and Translate")</sup>
- BahdanauAttention <sup>[*[11]*](#ref11 "Neural Machine Translation by Jointly Learning to Align and Translate")</sup>

`models.transformer_networks` [➜](src/models/transformer_networks.py)
- TransformerEncoderLayer
- TransformerEncoder
- TransformerDecoderLayer
- TransformerDecoder
- Transformer <sup>[*[12]*](#ref12 "Attention Is All You Need")</sup>

`models.blocks.convolutional_blocks` [➜](src/models/blocks/convolutional_blocks.py)
- Inception <sup>[*[18]*](#ref18 "Going deeper with convolutions")</sup>
- ResBlock <sup>[*[19]*](#ref19 "Deep Residual Learning for Image Recognition")</sup>
- ResBottleneckBlock <sup>[*[19]*](#ref19 "Deep Residual Learning for Image Recognition")</sup>
- ResNeXtBlock <sup>[*[20]*](#ref20 "Aggregated Residual Transformations for Deep Neural Networks")</sup>
- DenseLayer <sup>[*[22]*](#ref22 "Densely Connected Convolutional Networks")</sup>
- DenseBlock <sup>[*[22]*](#ref22 "Densely Connected Convolutional Networks")</sup>
- DenseTransition <sup>[*[22]*](#ref22 "Densely Connected Convolutional Networks")</sup>


### Example usages
- examples/ [➜](examples/)
- examples/convolutional [➜](examples/convolutional)
- examples/energy_based [➜](examples/energy_based)
- examples/graph [➜](examples/graph)
- examples/recurrent [➜](examples/recurrent)
- examples/attention [➜](examples/attention)
- examples/transformer [➜](examples/transformer)
- examples/shallow [➜](examples/shallow)


<hr/>


### References
1. <a name="ref1" href="https://proceedings.mlr.press/v37/ioffe15.pdf">Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift</a>
2. <a name="ref2" href="https://arxiv.org/pdf/1607.06450.pdf">Layer Normalization</a>
3. <a name="ref3" href="https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf">ImageNet Classification with Deep Convolutional Neural Networks</a>
4. <a name="ref4" href="https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf">Dropout: A Simple Way to Prevent Neural Networks from Overfitting</a>
5. <a name="ref5" href="https://arxiv.org/pdf/1308.0850.pdf">Generating Sequences With Recurrent Neural Networks</a>
6. <a name="ref6" href="https://arxiv.org/pdf/1709.01507.pdf">Squeeze-and-Excitation Gate layer</a>
7. <a name="ref7" href="https://arxiv.org/pdf/1609.02907.pdf">Semi-Supervised Classification with Graph Convolutional Networks</a>
8. <a name="ref8" href="https://arxiv.org/pdf/1706.02216.pdf">Inductive Representation Learning on Large Graphs</a>
9. <a name="ref9" href="https://proceedings.neurips.cc/paper_files/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf">Hierarchical Graph Representation Learning with Differentiable Pooling</a>
10. <a name="ref10" href="https://arxiv.org/pdf/1606.08415v5">Gaussian Error Linear Units (GELUs)</a>
11. <a name="ref11" href="https://arxiv.org/pdf/1409.0473.pdf">Neural Machine Translation by Jointly Learning to Align and Translate</a>
12. <a name="ref12" href="https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf">Attention Is All You Need</a>
13. <a name="ref13" href="https://arxiv.org/pdf/1301.3781.pdf">Efficient Estimation of Word Representations in Vector Space</a>
14. <a name="ref14" href="https://papers.nips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf">Sequence to Sequence Learning with Neural Networks</a>
15. <a name="ref15" href="https://hal.science/hal-03926082/document">Gradient-based learning applied to document recognition</a>
16. <a name="ref16" href="https://arxiv.org/pdf/1312.4400.pdf">Network In Network</a>
17. <a name="ref17" href="https://arxiv.org/pdf/1409.1556.pdf">Very Deep Convolutional Networks for Large-Scale Image Recognition</a>
18. <a name="ref18" href="https://arxiv.org/pdf/1409.4842.pdf?">Going deeper with convolutions</a>
19. <a name="ref19" href="https://arxiv.org/pdf/1512.03385.pdf">Deep Residual Learning for Image Recognition</a>
20. <a name="ref20" href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf">Aggregated Residual Transformations for Deep Neural Networks</a>
21. <a name="ref21" href="https://arxiv.org/pdf/1709.01507.pdf">Squeeze-and-Excitation Networks</a>
22. <a name="ref22" href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf">Densely Connected Convolutional Networks</a>
23. <a name="ref23" href="https://arxiv.org/pdf/1810.00826v3.pdf">How Powerful are Graph Neural Networks?</a>
24. <a name="ref24" href="https://arxiv.org/pdf/1406.6247.pdf">Recurrent Models of Visual Attention</a>
25. <a name="ref25" href="https://arxiv.org/pdf/1506.02025.pdf">Spatial Transformer Networks</a>

<!-- auto-generated-end -->

## Installation
```
conda env create --name dev --file=./environment.yml
```
