# tiny-grad
# TinyGrad From Scratch

A minimal implementation of automatic differentiation and backpropagation inspired by Andrej Karpathy's Micrograd.

## Overview

This project implements a tiny autograd engine from scratch using pure Python.

The goal is to understand:

* Computational graphs
* Forward propagation
* Backpropagation
* Automatic differentiation
* Gradient computation using the chain rule

Instead of operating on tensors like PyTorch, this implementation operates on scalar values, making the mechanics of backpropagation easy to understand.

---

## Features

* Custom `Value` class
* Computational graph construction
* Addition operation (`+`)
* Multiplication operation (`*`)
* Automatic gradient computation
* Reverse-mode autodifferentiation
* Topological sorting for graph traversal

---

## How Backpropagation Works

Consider:

```text
a = 2
b = -3
c = 10
f = 2

e = a * b
d = e + c
L = d * f
```

Computation graph:

```text
a ----\
       * --> e ----\
b ----/             +
                     --> d ----\
c ------------------/           *
                                 --> L
f ------------------------------/
```

Forward pass:

```text
e = -6
d = 4
L = 8
```

Backpropagation computes:

```text
dL/da = -6
dL/db = 4
dL/dc = 2
dL/df = 4
```

These gradients tell us how much each variable contributed to the final output.

---

## Mathematical Intuition

Backpropagation is an efficient application of the chain rule.

Example:

```text
L = (a*b + c)*f
```

To compute:

```text
dL/da
```

we use:

```text
dL/da =
(dL/dd) *
(dd/de) *
(de/da)
```

Instead of manually computing derivatives for every path, the computational graph stores local derivative rules and propagates gradients backward automatically.

---

## Running

```bash
python tiny-grad.py
```

Expected output:

```text
Loss: Value(data=8.0, grad=0.0)

a.grad = -6.0
b.grad = 4.0
c.grad = 2.0
f.grad = 4.0
```

---

## Project Structure

```text
.
├── tiny-grad.py
└── README.md
```

---

## Future Improvements

* Subtraction
* Division
* Power operator
* ReLU activation
* Tanh activation
* Neural network layers
* MLP implementation
* Tensor support
* Visualization of computation graphs

---

## Inspiration

This project is heavily inspired by:

* Andrej Karpathy's Micrograd
* Neural Networks: Zero to Hero series
* PyTorch Autograd

---

## Learning Outcome

After completing this project you should understand:

* What gradients are
* What the chain rule does
* How reverse-mode autodiff works
* How PyTorch computes gradients internally
* The core idea behind neural network training

---

## License

MIT License
