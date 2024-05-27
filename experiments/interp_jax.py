import interpax
import jax
import jax.numpy as jnp
import numpy as np


## 1D
xp = jnp.linspace(0.0, 2.0 * np.pi, 100)
xq = jnp.linspace(0.0, 2.0 * np.pi, 10000)
f = lambda x: jnp.sin(x)
fp = f(xp)

fq = interpax.interp1d(xq, xp, fp, method="cubic")
np.testing.assert_allclose(fq, f(xq), rtol=1e-6, atol=1e-5)


# 2D
x = jnp.linspace(0.0, 1.0, 100)
y = jnp.linspace(0.0, 1.0, 100)
f = np.ones((x.shape[0], y.shape[0]))

interpolator = interpax.Interpolator2D(
    x=x,
    y=y,
    f=f,
    method="linear",
    extrap=False,
    period=None,
)

points = jnp.zeros((1000, 2))
interpolator(points[:, 0], points[:, 1])
