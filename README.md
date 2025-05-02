# Regulating DINO-WM with Optimal Transport

This repository contains the implementation of our research on enhancing DINO-WM (World Models on Pre-trained Visual Features) with Optimal Transport regularization for improved zero-shot planning in robotics.

## Overview

We extend the DINO-WM framework introduced by Zhou et al. (2024) by incorporating Optimal Transport (OT) regularization to guide the planning process.
![Project Overview]

## Method

Our method builds upon DINO-WM, which leverages pre-trained DINOv2 visual features to create world models that can be used for zero-shot planning. Our approach constrains the world model's rollouts to follow shorter and more optimal trajectories by using OT interpolants as additional targets during planning. We do this by enhance the planning objective with an additional OT regularization term:

```
Cost = ||ẑT - zg||² + λ||ztOT - ẑt||²
```

Where:
- `ẑT` is the predicted latent state at the final timestep T
- `zg` is the goal latent state
- `ztOT` is the OT interpolant at timestep t
- `λ` is the regularization strength

The OT interpolant provides a direct path between the initial and goal states in the latent space, guiding the world model's predictions to follow more optimal trajectories.

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

Our OT-regularized approach shows improved planning performance compared to baseline methods, with significantly lower Chamfer distances in challenging manipulation tasks. The addition of OT regularization helps the world model find more direct paths to the goal while maintaining physical plausibility.

## Limitations

- Increased computational cost due to OT calculations
- Physically unrealistic interpolants in some cases
- Sensitivity to regularization strength parameter
- Potential failure cases where OT guidance is ineffective

## Installation and Usage

*Coming soon*


