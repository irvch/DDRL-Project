# Regulating DINO-WM with Optimal Transport

This repository contains the implementation of our research on enhancing DINO-WM (World Models on Pre-trained Visual Features) with Optimal Transport regularization for improved zero-shot planning in robotics.

## Overview

We extend the DINO-WM framework introduced by Zhou et al. (2024) by incorporating Optimal Transport (OT) regularization to guide the planning process.

## Method

Our method builds upon DINO-WM, which leverages pre-trained DINOv2 visual features to create world models that can be used for zero-shot planning. Our approach constrains the world model's rollouts to follow shorter and more optimal trajectories by using interpolants in Wasserstein space as additional subtargets during planning.

## Optimal Transport
We first compute the Optimal Transport (OT) map $T: X \to Y $ that pushes source points $x \in X$ to the target points $y\in Y$ such that the externally provided cost $c(x, T(x))$ of moving point $x$ to $T(x)$ is minimized. This idea is captured mathematically by
$\min_T \left\{\left.\int _{X}c(x,T(x))\,\mathrm{d} \mu(x)\;\right|\;T_{\#}(\mu )=\delta \right\}$

Here $T_{\#}(\mu) = \delta$ denotes a measure-preserving pushforward.

This project uses the Euclidean distance, a canonical choice of cost in normed spaces:
$c(x,T(x)) = \frac{1}{2n} \sum_{i=1}^n ||x_i - T(x_i)||^2_2$
where $n$ is the number of observations and $d$ is the number of dimensions.

Test Function
TO enforce the push-forward condition, we regulate transport with the Kullback-Liebler divergence - measuring relative entropy between the source distribution and the target distribution.
$D_{KL}(\delta||\mu) = \int_{\mathbb{R}^n} \mu(x) \ln\frac{\delta(x)}{\mu(x)} dx \approx \sum_{i=1}^n \ln\frac{\delta(x_i)}{\mu(x_i)}$

We employ kernel density estimation to acquire our probability density functions
\begin{equation}
    \delta(x_i)=\frac{1}{n}\sum_{j=1}^n K(T(x_i),T(x_j),H_x)
\end{equation}
\begin{equation}
    \mu(x_i)=\frac{1}{m}\sum_{k=1}^m K(T(x_i),y_k,H_y)
\end{equation}

Here we use the multivariate Gaussian kernel:
\begin{equation}
    K(x,\mu,\Sigma)=\frac{1}{\sqrt{(2\pi)^d det(\Sigma)}} e^{-\frac{1}{2}(x-\mu) \Sigma^{-1} (x-\mu)^T }
\end{equation}
where $\Sigma$ represents the diagonal covariance matrix, and $\mu$ denotes the center point of the kernel.

To find the optimal transport map, we seek to minimize the global cost function 
$L(x, y) = c(x,T(x)) - \lambda \left[ D_{KL}(\delta||\mu) \right]$


### Interpolation in Wasserstein Space

We generate intermediate waypoints between the source and target states:
For patch-based features:
- We calculate the transport-guided target for each patch using the OT plan: $z_{mapped} = P \cdot z_{target}$
- For each timestep $t$, we interpolate: $\text{waypoint}[t] = (1-\alpha) \cdot z_{source} + \alpha \cdot z_{mapped}$
- where $\alpha = t/(num\_steps+1)$ controls the interpolation ratio

The OT interpolant provides a direct path between the initial and goal states in the latent space, guiding the world model's predictions to follow more optimal trajectories.

We do this by enhance the planning objective with an additional OT regularization term:

$\text{Cost} = \|\hat{z}_T - z_g\|^2 + \lambda\|z_t^{OT} - \hat{z}_t\|^2$

Where:
- $\hat{z}_T$ is the predicted latent state at the final timestep T
- $z_g$ is the goal latent state
- $z_t^{OT}$ is the OT interpolant at timestep t
- $\lambda$ is the regularization strength


## Model Predictive Control

We use the Cross-Entropy Method (CEM) for Model Predictive Control:

1. Encode current and goal observations into latent states
2. Sample initial action sequences
3. Predict future latents using the world model
4. Score sequences using our enhanced OT-regularized cost function
5. Use CEM to refine the action distribution
6. Execute the first action of the best sequence
7. Repeat the process

## Results

We evaluate the OT-regularized planning method on the PushT environment. Our approach demonstrates slight improvements in performance compared to baseline methods from the original DINO-WM implementation. We can note that the addition of OT regularization helps the world model find more direct paths to the goal - rounding corners more tightly and traversing more quickly.

*Figures coming soon*

## Limitations
There are considerable drawbacks to this implementation. Chief among them is the increased computational cost due to optimal transport calculations being made at every initial planning step. Since 
- Physically unrealistic interpolants in some cases
- Sensitivity to regularization strength parameter
- Potential failure cases where OT guidance is ineffective
Coupled with only marginal gains in the cases where it is sucessful. The method overall seems mostly ineffective on the short-range tasks included in these experiments.
