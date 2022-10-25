from typing import Any, Optional, Union, Callable

import jax
from jax import numpy as jnp, tree_util
from jax.numpy.linalg import norm
from optax._src import combine, base, numerics, utils, transform
from optax._src.alias import ScalarOrSchedule, _scale_by_learning_rate
from optax._src.transform import bias_correction, update_moment, update_moment_per_elem_norm, ScaleByAdamState

def scale_by_edam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Edam algorithm.

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
      lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b1, 1)
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)

    def update(m, v):
      d = jnp.sqrt(v) + jnp.abs(m) * eps
      return m / jnp.where(d, d, jnp.inf)

    updates = jax.tree_util.tree_map(update, mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def edam(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  return combine.chain(
    scale_by_edam(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype),
    _scale_by_learning_rate(learning_rate),
  )


def edamw(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  return combine.chain(
    scale_by_edam(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype),
    transform.add_decayed_weights(weight_decay, mask),
    _scale_by_learning_rate(learning_rate),
  )

def scale_by_edam2(
    b1: float = 0.9,
    b2: float = 0.999,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Edam algorithm.

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
      lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):

    del params
    mu = update_moment(updates, state.mu, b1, 1)
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)

    param_count = sum(tree_util.tree_leaves(jax.tree_map(lambda x: x.size, updates)))
    mu_abs_mean = sum(tree_util.tree_leaves(jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), updates))) / param_count

    def update(m, v):
      d = jnp.sqrt(v) * mu_abs_mean
      return jnp.abs(m) * m / jnp.where(d, d, jnp.inf)

    updates = jax.tree_util.tree_map(update, mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)

def edam2(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 0,  # ignore
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  return combine.chain(
    scale_by_edam2(b1=b1, b2=b2, mu_dtype=mu_dtype),
    _scale_by_learning_rate(learning_rate),
  )


def edam2w(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 0,  # ignore
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  return combine.chain(
    scale_by_edam2(b1=b1, b2=b2, mu_dtype=mu_dtype),
    transform.add_decayed_weights(weight_decay, mask),
    _scale_by_learning_rate(learning_rate),
  )
