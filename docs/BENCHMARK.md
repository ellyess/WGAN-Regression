# Benchmark results

Produced by `scripts/run_benchmark.py`. Each model draws one y
sample per test input; metrics compare those samples with the true
test set (lower is better for all three).

- **W1**: mean Wasserstein-1 distance between conditional slices
  of p(y | x)
- **MMD**: kernel maximum mean discrepancy between the joint
  (x, y) samples
- **NLL**: negative log-likelihood of the true test points under
  a KDE of the model samples

| Scenario | Model | W1 | MMD | NLL |
|---|---|---|---|---|
| sinus | WGAN-GP | 0.502 | 0.0159 | 3.44 |
|  | GPR | **0.054** | 0.0001 | 2.70 |
|  | MDN | 0.064 | **0.0000** | 2.71 |
|  | Diffusion | 0.087 | 0.0005 | **2.69** |
| circle | WGAN-GP | **0.150** | 0.0046 | 1.24 |
|  | GPR | 0.258 | 0.0074 | 1.56 |
|  | MDN | 0.250 | 0.0017 | **1.17** |
|  | Diffusion | 0.206 | **0.0017** | 1.18 |
| multi | WGAN-GP | 0.402 | 0.0961 | 6.51 |
|  | GPR | **0.177** | 0.0141 | 0.83 |
|  | MDN | 0.299 | 0.0039 | 0.67 |
|  | Diffusion | 0.181 | **0.0035** | **0.60** |
| moons | WGAN-GP | 0.267 | 0.0192 | 1.64 |
|  | GPR | 0.216 | 0.0065 | 1.70 |
|  | MDN | **0.102** | 0.0012 | **1.45** |
|  | Diffusion | 0.165 | **0.0007** | 1.52 |
| heter | WGAN-GP | 0.185 | 0.0260 | 2.06 |
|  | GPR | 0.192 | 0.0114 | 0.65 |
|  | MDN | 0.075 | **0.0016** | 0.40 |
|  | Diffusion | **0.073** | 0.0018 | **0.39** |
| 3d | WGAN-GP | 0.167 | 0.0004 | 5.42 |
|  | GPR | **0.015** | 0.0000 | 5.40 |
|  | MDN | 0.024 | **0.0000** | 5.40 |
|  | Diffusion | 0.063 | 0.0000 | **5.39** |
| helix | WGAN-GP | 2.001 | 0.0049 | 5.61 |
|  | GPR | 1.781 | 0.0136 | 5.75 |
|  | MDN | **1.567** | 0.0045 | **5.35** |
|  | Diffusion | 1.581 | **0.0013** | 5.57 |
