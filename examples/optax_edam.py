from typing import Any, Optional, NamedTuple, Union, Callable

import chex
import jax
from jax import numpy as jnp
from optax._src import combine, base, numerics, utils, transform
from optax._src.alias import ScalarOrSchedule, _scale_by_learning_rate
from optax._src.transform import bias_correction, update_moment, update_moment_per_elem_norm


class ScaleByEdamState(NamedTuple):
  """State for the Edam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates

def scale_by_edam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Adam algorithm.

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
    return ScaleByEdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b1, 1)
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)

    def update(m, v):
      d = jnp.sqrt(v) + jnp.abs(m) * eps
      return m / jnp.where(jnp.equal(d, 0), 1, d)

    updates = jax.tree_util.tree_map(update, mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
    return updates, ScaleByEdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def edam(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """The classic Edam optimiser.

  Edam is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is computed from estimates of first- and second-order
  moments of the gradients (using suitable exponential moving averages).

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Edam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for example when computing (meta-)gradients through Edam.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    the corresponding `GradientTransformation`.
  """
  return combine.chain(
    scale_by_edam(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype),
    _scale_by_learning_rate(learning_rate),
  )

def edamw(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Adam with weight decay regularization.

  AdamW uses weight decay to regularise learning towards small weights, as
  this leads to better generalisation. In SGD you can also use L2 regularisation
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam.

  WARNING: Sometimes you may want to skip weight decay for BatchNorm scale or
  for the bias parameters. You can use `optax.masked` to make your own AdamW
  variant where `additive_weight_decay` is applied only to a subset of `params`.

  References:
    Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for instance when computing (meta-)gradients through Adam.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: a tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.

  Returns:
    the corresponding `GradientTransformation`.
  """
  return combine.chain(
      scale_by_edam(
          b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype),
      transform.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
  )