Spiking Neural Network with Class-Incremental Learning
A from-scratch implementation of a spiking neural network (SNN) with surrogate gradient training and class-incremental learning on CIFAR-10. Built entirely without AI assistance to develop a deep understanding of neuromorphic computing fundamentals.
What This Demonstrates

Custom LIF neurons — Leaky Integrate-and-Fire neuron model with configurable threshold and membrane decay (β), implementing the dynamics V[t] = β · V[t-1] + x with soft reset after spike emission
Surrogate gradient backpropagation — Custom torch.autograd.Function using an arctan-based surrogate to enable gradient flow through the non-differentiable spike function
Spiking CNN architecture — 3-layer convolutional network (64 → 128 → 256 channels) with batch normalization, average pooling, and temporal spike accumulation over 10 timesteps
Class-incremental learning (CIL) — Dynamic output layer expansion that preserves learned weights when new classes are introduced, trained sequentially on 5 tasks of 2 classes each
Full training pipeline — Data loading, task splitting, training loop, and per-task evaluation in a single self-contained file

Architecture
Input (3×32×32)
  → Conv2d(3→64) → BatchNorm → LIF
  → Conv2d(64→128) → BatchNorm → AvgPool → LIF
  → Conv2d(128→256) → BatchNorm → LIF → AvgPool
  → Flatten → Linear(256×8×8 → num_classes) → LIF
  → Spike accumulation over 10 timesteps → Mean output
How It Works
Spiking neurons replace traditional activation functions. Instead of outputting continuous values, each LIF neuron accumulates input into a membrane potential and emits a binary spike (0 or 1) when the potential exceeds a threshold. The membrane potential decays by factor β each timestep, mimicking biological neuron dynamics.
Surrogate gradients solve the non-differentiability problem. Since the spike function has zero gradient almost everywhere, the backward pass substitutes an arctan-based smooth approximation, allowing standard backpropagation through the spiking layers.
Class-incremental learning trains the network on 2 new classes at a time (5 tasks total for CIFAR-10's 10 classes). When new classes arrive, the output layer expands while preserving existing weights — the network must learn new classes without catastrophically forgetting old ones.
Usage
bashpip install torch torchvision
python snn.py
The script automatically downloads CIFAR-10, trains sequentially on tasks [[0,1], [2,3], [4,5], [6,7], [8,9]], and evaluates accuracy on all classes seen so far after each task.
Requirements

Python 3.8+
PyTorch
torchvision
CUDA-capable GPU (optional, falls back to CPU)

Tech Stack
ComponentImplementationNeuron modelLeaky Integrate-and-Fire (LIF) with soft resetGradient methodArctan surrogate gradient via torch.autograd.FunctionArchitecture3-layer spiking CNN + linear classifierLearning paradigmClass-incremental learning with output expansionDatasetCIFAR-10 (5 tasks × 2 classes)FrameworkPyTorch
Author
Joshua Blaszczyk
