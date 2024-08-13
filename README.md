## Deep learning from scratch

> "What I cannot create, I do not understand." -- Richard Feynman

I agree.


---
Clean code implementation of the foundational deep learning layers, optimizers and models
* using PyTorch's autograd for the backpropagation
* using PyTorch's tensors for GPU computation
---

<!-- auto-generated-start -->


### Layers

`lib.layers` [➜](src/lib/layers.py)
- Linear
- Embedding
- BatchNorm <sup>[*[1]*](#ref1 "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift")</sup>
- BatchNorm1d
- BatchNorm2d
- LayerNorm <sup>[*[2]*](#ref2 "Layer Normalization")</sup>
- RMSNorm <sup>[*[3]*](#ref3 "Root Mean Square Layer Normalization")</sup>
- LocalResponseNorm <sup>[*[4]*](#ref4 "ImageNet Classification with Deep Convolutional Neural Networks")</sup>
- Dropout <sup>[*[5]*](#ref5 "Dropout: A Simple Way to Prevent Neural Networks from Overfitting")</sup>
- RNN_cell
- LSTM_cell <sup>[*[6]*](#ref6 "Generating Sequences With Recurrent Neural Networks")</sup>
- GRU_cell
- RNN
- Conv2d
- Conv2dGroups
- Pool2d
- MaxPool2d
- AvgPool2d
- BatchAddPool
- SEGate <sup>[*[7]*](#ref7 "Squeeze-and-Excitation Gate layer")</sup>
- Graph_cell
- GCN_cell <sup>[*[8]*](#ref8 "Semi-Supervised Classification with Graph Convolutional Networks")</sup>
- GraphSAGE_cell <sup>[*[9]*](#ref9 "Inductive Representation Learning on Large Graphs")</sup>
- DiffPool <sup>[*[10]*](#ref10 "Hierarchical Graph Representation Learning with Differentiable Pooling")</sup>
- ReLU
- GELU <sup>[*[11]*](#ref11 "Gaussian Error Linear Units (GELUs)")</sup>
- GLU <sup>[*[12]*](#ref12 "Language Modeling with Gated Convolutional Networks")</sup>
- SwiGLU <sup>[*[13]*](#ref13 "GLU Variants Improve Transformer")</sup>
- Flatten
- DotProductAttention
- AdditiveAttention <sup>[*[14]*](#ref14 "Neural Machine Translation by Jointly Learning to Align and Translate")</sup>
- DiagBlockAttention <sup>[*[15]*](#ref15 "Generating Long Sequences with Sparse Transformers")</sup>
- ColumnBlockAttention <sup>[*[15]*](#ref15 "Generating Long Sequences with Sparse Transformers")</sup>
- MultiHeadAttention <sup>[*[16]*](#ref16 "Attention Is All You Need")</sup>
- GroupedQueryAttention <sup>[*[17]*](#ref17 "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints")</sup>
- RelativeWindowAttention
- SparseMultiHeadAttention <sup>[*[15]*](#ref15 "Generating Long Sequences with Sparse Transformers")</sup>
- KVCache
- PositionalEncoding <sup>[*[16]*](#ref16 "Attention Is All You Need")</sup>
- RotaryEncoding <sup>[*[18]*](#ref18 "RoFormer: Enhanced Transformer with Rotary Position Embedding")</sup>
- PatchEmbedding <sup>[*[19]*](#ref19 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale")</sup>
- RelativePositionBias2d <sup>[*[20]*](#ref20 "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows")</sup>

`lib.autoencoders` [➜](src/lib/autoencoders.py)
- MatrixFactorization
- AutoencoderLinear
- Word2Vec <sup>[*[21]*](#ref21 "Efficient Estimation of Word Representations in Vector Space")</sup>


### Optimizers

`lib.optimizers` [➜](src/lib/optimizers.py)
- Optimizer
- SGD
- SGD_Momentum
- AdaGrad
- RMSProp
- AdaDelta
- Adam
- AdamW <sup>[*[22]*](#ref22 "Decoupled Weight Decay Regularization")</sup>
- LR_Scheduler
- LR_StepScheduler
- LR_PlateauScheduler
- LR_CosineDecayScheduler


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
- LSTM <sup>[*[6]*](#ref6 "Generating Sequences With Recurrent Neural Networks")</sup>
- GRU
- LangModel
- EchoStateNetwork
- Encoder
- Decoder
- Seq2Seq <sup>[*[23]*](#ref23 "Sequence to Sequence Learning with Neural Networks")</sup>

`models.convolutional_networks` [➜](src/models/convolutional_networks.py)
- SimpleCNN
- SimpleFullyCNN
- LeNet5 <sup>[*[24]*](#ref24 "Gradient-based learning applied to document recognition")</sup>
- AlexNet <sup>[*[4]*](#ref4 "ImageNet Classification with Deep Convolutional Neural Networks")</sup>
- NetworkInNetwork <sup>[*[25]*](#ref25 "Network In Network")</sup>
- VGG16 <sup>[*[26]*](#ref26 "Very Deep Convolutional Networks for Large-Scale Image Recognition")</sup>
- GoogLeNet <sup>[*[27]*](#ref27 "Going deeper with convolutions")</sup>
- DeepPlainCNN

`models.residual_networks` [➜](src/models/residual_networks.py)
- ResNet34 <sup>[*[28]*](#ref28 "Deep Residual Learning for Image Recognition")</sup>
- ResNet50 <sup>[*[28]*](#ref28 "Deep Residual Learning for Image Recognition")</sup>
- ResNeXt50 <sup>[*[29]*](#ref29 "Aggregated Residual Transformations for Deep Neural Networks")</sup>
- SEResNet50 <sup>[*[30]*](#ref30 "Squeeze-and-Excitation Networks")</sup>
- SEResNeXt50 <sup>[*[30]*](#ref30 "Squeeze-and-Excitation Networks")</sup>
- DenseNet121 <sup>[*[31]*](#ref31 "Densely Connected Convolutional Networks")</sup>

`models.graph_networks` [➜](src/models/graph_networks.py)
- GCN <sup>[*[8]*](#ref8 "Semi-Supervised Classification with Graph Convolutional Networks")</sup>
- GraphSAGE <sup>[*[9]*](#ref9 "Inductive Representation Learning on Large Graphs")</sup>
- GIN <sup>[*[32]*](#ref32 "How Powerful are Graph Neural Networks?")</sup>
- DiffPoolNet <sup>[*[10]*](#ref10 "Hierarchical Graph Representation Learning with Differentiable Pooling")</sup>

`models.attention_networks` [➜](src/models/attention_networks.py)
- RecurrentAttention <sup>[*[33]*](#ref33 "Recurrent Models of Visual Attention")</sup>
- SpatialTransformer <sup>[*[34]*](#ref34 "Spatial Transformer Networks")</sup>
- SpatialTransformerNet
- AttentionEncoder <sup>[*[14]*](#ref14 "Neural Machine Translation by Jointly Learning to Align and Translate")</sup>
- AttentionDecoder <sup>[*[14]*](#ref14 "Neural Machine Translation by Jointly Learning to Align and Translate")</sup>
- BahdanauAttention <sup>[*[14]*](#ref14 "Neural Machine Translation by Jointly Learning to Align and Translate")</sup>

`models.transformer_networks` [➜](src/models/transformer_networks.py)
- TransformerEncoderLayer
- TransformerEncoder
- TransformerDecoderLayer
- TransformerDecoder
- Transformer <sup>[*[16]*](#ref16 "Attention Is All You Need")</sup>
- GPT_TransformerBlock <sup>[*[35]*](#ref35 "Language Models are Unsupervised Multitask Learners")</sup>
- GPT_SparseTransformerBlock <sup>[*[15]*](#ref15 "Generating Long Sequences with Sparse Transformers")</sup>
- GPT2 <sup>[*[35]*](#ref35 "Language Models are Unsupervised Multitask Learners")</sup>
- GPT3 <sup>[*[36]*](#ref36 "Language Models are Few-Shot Learners")</sup>
- LLaMA_TransformerBlock <sup>[*[37]*](#ref37 "LLaMA: Open and Efficient Foundation Language Models")</sup>
- LLaMA1 <sup>[*[37]*](#ref37 "LLaMA: Open and Efficient Foundation Language Models")</sup>
- LLaMA2 <sup>[*[38]*](#ref38 "Llama 2: Open Foundation and Fine-Tuned Chat Models")</sup>

`models.visual_transformers` [➜](src/models/visual_transformers.py)
- VisionTransformer <sup>[*[19]*](#ref19 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale")</sup>
- VisionTransformerConvStem <sup>[*[39]*](#ref39 "Early Convolutions Help Transformers See Better")</sup>
- SwinTransformerBlock <sup>[*[20]*](#ref20 "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows")</sup>
- SwinTransformer <sup>[*[20]*](#ref20 "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows")</sup>

`models.blocks.convolutional_blocks` [➜](src/models/blocks/convolutional_blocks.py)
- Inception <sup>[*[27]*](#ref27 "Going deeper with convolutions")</sup>
- ResBlock <sup>[*[28]*](#ref28 "Deep Residual Learning for Image Recognition")</sup>
- ResBottleneckBlock <sup>[*[28]*](#ref28 "Deep Residual Learning for Image Recognition")</sup>
- ResNeXtBlock <sup>[*[29]*](#ref29 "Aggregated Residual Transformations for Deep Neural Networks")</sup>
- DenseLayer <sup>[*[31]*](#ref31 "Densely Connected Convolutional Networks")</sup>
- DenseBlock <sup>[*[31]*](#ref31 "Densely Connected Convolutional Networks")</sup>
- DenseTransition <sup>[*[31]*](#ref31 "Densely Connected Convolutional Networks")</sup>


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
3. <a name="ref3" href="https://arxiv.org/pdf/1910.07467">Root Mean Square Layer Normalization</a>
4. <a name="ref4" href="https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf">ImageNet Classification with Deep Convolutional Neural Networks</a>
5. <a name="ref5" href="https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf">Dropout: A Simple Way to Prevent Neural Networks from Overfitting</a>
6. <a name="ref6" href="https://arxiv.org/pdf/1308.0850.pdf">Generating Sequences With Recurrent Neural Networks</a>
7. <a name="ref7" href="https://arxiv.org/pdf/1709.01507.pdf">Squeeze-and-Excitation Gate layer</a>
8. <a name="ref8" href="https://arxiv.org/pdf/1609.02907.pdf">Semi-Supervised Classification with Graph Convolutional Networks</a>
9. <a name="ref9" href="https://arxiv.org/pdf/1706.02216.pdf">Inductive Representation Learning on Large Graphs</a>
10. <a name="ref10" href="https://proceedings.neurips.cc/paper_files/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf">Hierarchical Graph Representation Learning with Differentiable Pooling</a>
11. <a name="ref11" href="https://arxiv.org/pdf/1606.08415v5">Gaussian Error Linear Units (GELUs)</a>
12. <a name="ref12" href="https://arxiv.org/pdf/1612.08083v3">Language Modeling with Gated Convolutional Networks</a>
13. <a name="ref13" href="https://arxiv.org/pdf/2002.05202">GLU Variants Improve Transformer</a>
14. <a name="ref14" href="https://arxiv.org/pdf/1409.0473.pdf">Neural Machine Translation by Jointly Learning to Align and Translate</a>
15. <a name="ref15" href="https://arxiv.org/pdf/1904.10509">Generating Long Sequences with Sparse Transformers</a>
16. <a name="ref16" href="https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf">Attention Is All You Need</a>
17. <a name="ref17" href="https://arxiv.org/pdf/2305.13245v3">GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints</a>
18. <a name="ref18" href="https://arxiv.org/pdf/2104.09864">RoFormer: Enhanced Transformer with Rotary Position Embedding</a>
19. <a name="ref19" href="https://arxiv.org/pdf/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a>
20. <a name="ref20" href="https://arxiv.org/pdf/2103.14030">Swin Transformer: Hierarchical Vision Transformer using Shifted Windows</a>
21. <a name="ref21" href="https://arxiv.org/pdf/1301.3781.pdf">Efficient Estimation of Word Representations in Vector Space</a>
22. <a name="ref22" href="https://arxiv.org/pdf/1711.05101">Decoupled Weight Decay Regularization</a>
23. <a name="ref23" href="https://papers.nips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf">Sequence to Sequence Learning with Neural Networks</a>
24. <a name="ref24" href="https://hal.science/hal-03926082/document">Gradient-based learning applied to document recognition</a>
25. <a name="ref25" href="https://arxiv.org/pdf/1312.4400.pdf">Network In Network</a>
26. <a name="ref26" href="https://arxiv.org/pdf/1409.1556.pdf">Very Deep Convolutional Networks for Large-Scale Image Recognition</a>
27. <a name="ref27" href="https://arxiv.org/pdf/1409.4842.pdf?">Going deeper with convolutions</a>
28. <a name="ref28" href="https://arxiv.org/pdf/1512.03385.pdf">Deep Residual Learning for Image Recognition</a>
29. <a name="ref29" href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf">Aggregated Residual Transformations for Deep Neural Networks</a>
30. <a name="ref30" href="https://arxiv.org/pdf/1709.01507.pdf">Squeeze-and-Excitation Networks</a>
31. <a name="ref31" href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf">Densely Connected Convolutional Networks</a>
32. <a name="ref32" href="https://arxiv.org/pdf/1810.00826v3.pdf">How Powerful are Graph Neural Networks?</a>
33. <a name="ref33" href="https://arxiv.org/pdf/1406.6247.pdf">Recurrent Models of Visual Attention</a>
34. <a name="ref34" href="https://arxiv.org/pdf/1506.02025.pdf">Spatial Transformer Networks</a>
35. <a name="ref35" href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">Language Models are Unsupervised Multitask Learners</a>
36. <a name="ref36" href="https://arxiv.org/pdf/2005.14165">Language Models are Few-Shot Learners</a>
37. <a name="ref37" href="https://arxiv.org/pdf/2302.13971">LLaMA: Open and Efficient Foundation Language Models</a>
38. <a name="ref38" href="https://arxiv.org/pdf/2307.09288">Llama 2: Open Foundation and Fine-Tuned Chat Models</a>
39. <a name="ref39" href="https://arxiv.org/pdf/2106.14881">Early Convolutions Help Transformers See Better</a>

<!-- auto-generated-end -->

## Installation
### Local Setup
```
conda env create --name dev --file=./environment.yml
```
### Docker Setup
```
docker build -t deep . 
docker run --rm --gpus all --name deep deep 
docker exec -it deep /bin/bash
```
```
# For debugging use:
docker run -v .:/deep-learning --rm --gpus all --name deep deep
```
