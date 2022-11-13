# Clove

Clove is an automatic differentiation engine with a pytorch autograd like api and the ability to toggle numerical computation backends (currently implements `numpy` and `cupy`). It supports higher order backward-mode (vjp), forward-mode (jvp) and mixed mode autodiff.

Clove also implements a jax like functional differentiation module to take derivatives of arbitrary python functions.

Computations can be performed with specific backends :

```py
from clove import numpy as cnp

x = cnp.ones((2,2), requires_grad = True)
y = cnp.sum(x)
y.backward()
```

Or directly from the clove namespace (with the current backend): 

```py
import clove

x = clove.ones(2,2, requires_grad = True)
y = clove.sum(x)
y.backward()
```

the default backend is `'numpy'` and can be toggled using `clove.set_backend('backend_name')`

Clove has inbuilt `graphviz` extensions to visualize your computation graphs as you build them:

```py
import clove

a = clove.ones(3,1,1 requires_grad = True, name='a')
b = clove.randn(3,1,1 requires_grad = True, name='b')
c = a / b
d = a * b
e = clove.exp(d)
f = c.log()
g = e + f
h = g.sum()
clove.make_dot(h)
```

Like JAX, grads of functions can be taken just as easily:

```py
import clove

sigmoid = lambda x: 1/(1+clove.exp(-x))

sigmoid_grad = clove.grad(sigmoid)

print(sigmoid_grad(2.0))
```

