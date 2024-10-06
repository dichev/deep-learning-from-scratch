# Example usage of graph models


<br>
<details>
    <summary>
        <code>python <a href="graph_to_node_classification_simple.py">graph_to_node_classification_simple.py</a></code>
    </summary>

```text
$ python examples/graph/graph_to_node_classification_simple.py
GraphNet(): 592 parameters
Epoch 100 | Loss: 0.00 | Acc: 100.00%

GCN(): 316 parameters
Epoch 100 | Loss: 0.13 | Acc: 100.00%

GraphSAGE(): 2,496 parameters
Epoch 100 | Loss: 0.17 | Acc: 100.00%
```
</details>
<img src="../../docs/images/graph__graph_to_node_classification_simple.png" align="right" width="20%">
<h3>Classify nodes in a graph</h3>
<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://math.vercel.app/?color=black&from=h_i=\mathrm{relu}\left(\mathbf%20W_1x_i%20%2B%20\mathbf%20W_2\text{%20AGGREG}(\{x_j\}_{\mathcal%20N})\right)">
  <img alt="h_i=\mathrm{relu}\left(\mathbf W_2\text{ AGGREGATE}(\{x_j\}_{\mathcal N})+\mathbf W_1x_i\right)" src="https://math.vercel.app/?color=white&from=h_i=\mathrm{relu}\left(\mathbf%20W_1x_i%20%2B%20\mathbf%20W_2\text{%20AGGREG}(\{x_j\}_{\mathcal%20N})\right)">
</picture>
<br>

- GraphNet (Neighbourhood)
- GraphConv (Convolutional)
- GraphSAGE (Sample and Aggregate, with maxpool)


<br>

---


<br>
<details>
    <summary>
        <code>python <a href="graph_to_graph_classification.py">graph_to_graph_classification.py</a></code>
    </summary>

```text
$ python examples/graph/graph_to_graph_classification.py
Dataset: PROTEINS(1101)
Number of graphs: 1101
Number of nodes: 200
Number of features: 3
Number of classes: 2

GIN(): 59,266 parameters
Epoch 100 | Train Total Loss: 0.46 | Train Acc: 79.58% | Val Acc: 48.44%
Test Acc: 51.72%

DiffPoolNet(): 47,260 parameters
Epoch 100 | Train Total Loss: 3.48 | Train Acc: 80.47% | Val Acc: 52.34%
Test Acc: 64.53%

```
</details>
<img src="../../docs/images/graph__graph_to_graph_classification.png" align="right" width="20%">
<h3>Classify protein graphs</h3>

- GIN (Graph Isomorphism Network)

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://math.vercel.app/?color=black&from=H%27=f_{embed}(G)%20=\mathrm%20{MLP}\left(%20(1%2B\epsilon)X%20%2B%20(A-I)X%20%20\right)">
  <img alt="H'=f_{embed}(G) =\mathrm {MLP}\left((1+\epsilon)X + (A-I)X \right)" src="https://math.vercel.app/?color=white&from=H%27=f_{embed}(G)%20=\mathrm%20{MLP}\left(%20(1%2B\epsilon)X%20%2B%20(A-I)X%20%20\right)">
</picture>


- DiffPool (Pooling with learnable assignment, for hierarchical representations)

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://math.vercel.app/?color=black&from=G%27%20=%20S^Tf_{embed}(G)%2C%20\quad%20\text{%20where%20}%20S=\mathrm{softmax}(f_{pool}(G))">
  <img alt="G' = S^Tf_{embed}(G), \quad \text{ where } S=\mathrm{softmax}(f_{pool}(G))" src="https://math.vercel.app/?color=white&from=G%27%20=%20S^Tf_{embed}(G)%2C%20\quad%20\text{%20where%20}%20S=\mathrm{softmax}(f_{pool}(G))">
</picture>
<br>

