# Example usage of diffusion process


<details>
    <summary>
        <code>python <a href="generate_colored_mnist_diffusion.py">generate_colored_mnist_diffusion.py</a></code>
    </summary>

```text
$ python examples/diffusion/generate_colored_mnist_diffusion.py
Epoch 50/50: 100%|██████████| 469/469 [loss=0.0041 avg_loss=0.0049]
Generating 10 images with context tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])..
```
</details>
<img src="../../docs/images/diffusion__generate_colored_mnist_diffusion.png" align="right" width="20%">
<h3>Generate colored digits</h3>

The **forward (diffusion) process** is gradually adding Gaussian noise to the image at each timestamp:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://math.vercel.app/?color=black&from=q(x_{0:T})%20=%20\underbrace{q(x_0)}_{\text{Image}}%20\prod%20\underbrace%20{q(x_t|x_{t-1})}_{\mathcal%20N_t}">
  <img alt="q(x_{0:T}) = \underbrace{q(x_0)}_{\text{Image}} \prod \underbrace {q(x_t|x_{t-1})}_{\mathcal N_t}" src="https://math.vercel.app/?color=white&from=q(x_{0:T})%20=%20\underbrace{q(x_0)}_{\text{Image}}%20\prod%20\underbrace%20{q(x_t|x_{t-1})}_{\mathcal%20N_t}">
</picture>

The **reverse process** is learning the Gaussian transitions to restore the image:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://math.vercel.app/?color=black&from=p(x_{0:T})%20=%20\underbrace{%20p(x_T)%20}_{\mathcal%20N(\mathbf%200%2C\%2C\mathbf%20I)}%20\prod%20\underbrace{p_{\theta}%20(x_{t-1}|x_{t})}_{\mathcal%20N_t(\mu_\theta%2C\%2C\sigma^2\mathbf%20I)}">
  <img alt="p(x_{0:T}) = \underbrace{ p(x_T) }_{\mathcal N(\mathbf 0,\,\mathbf I)} \prod \underbrace{p_{\theta} (x_{t-1}|x_{t})}_{\mathcal N_t(\mu_\theta,\,\sigma^2\mathbf I)}" src="https://math.vercel.app/?color=white&from=p(x_{0:T})%20=%20\underbrace{%20p(x_T)%20}_{\mathcal%20N(\mathbf%200%2C\%2C\mathbf%20I)}%20\prod%20\underbrace{p_{\theta}%20(x_{t-1}|x_{t})}_{\mathcal%20N_t(\mu_\theta%2C\%2C\sigma^2\mathbf%20I)}">
</picture>
<br><br><br><br><br><br>

