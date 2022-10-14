# Clove

Clove is an automatic differentiation engine with a pytorch autograd like api and the ability to toggle numerical computation backends (currently implements numpy and cupy). It supports higher order backward-mode (vjp), forward-mode (jvp) and mixed mode autodiff.

Clove also implements a jax like functional differentiation module to take derivatives of arbitrary python functions.
