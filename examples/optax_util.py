from functools import partial

import optax
import wandb
from jax import jit, numpy as jnp, tree_util
from optax._src.transform import bias_correction

from examples import optax_edam


def _tree_mean(tree):
  leaves = tree_util.tree_leaves(tree)
  count = sum([x.size for x in leaves])
  return jnp.sum(jnp.array([jnp.sum(x) for x in leaves])) / count


def _tree_flatten(tree):
  return jnp.concatenate([jnp.reshape(x, (-1,)) for x in tree_util.tree_leaves(tree)])


@partial(jit, static_argnames=['b1', 'b2'])
def _stats(state, b1, b2):
  opt_state = state.opt_state[0]
  if not hasattr(opt_state, 'mu'):
    return dict()

  abs_m = _tree_flatten(tree_util.tree_map(
    lambda x: jnp.abs(bias_correction(x, b1, opt_state.count + 1)),
    opt_state.mu))

  sqrt_v = _tree_flatten(tree_util.tree_map(
    lambda x: jnp.sqrt(bias_correction(x, b2, opt_state.count + 1)),
    opt_state.nu))

  log_hist = lambda x: jnp.histogram(jnp.nan_to_num(jnp.log(x), neginf=-88), bins=128)
  to_log10_hist = lambda log_hist: (log_hist[0], log_hist[1] / jnp.log(10))

  abs_m_log_hist = log_hist(abs_m)
  sqrt_v_log_hist = log_hist(sqrt_v)
  return dict(
    grad_abs_log_hist=abs_m_log_hist,
    grad_abs_log10_hist=to_log10_hist(abs_m_log_hist),
    grad_abs_mean=jnp.mean(abs_m),
    grad_abs_var=jnp.var(abs_m),
    sqrt_v_log_hist=sqrt_v_log_hist,
    sqrt_v_log10_hist=to_log10_hist(sqrt_v_log_hist),
    sqrt_v_mean=jnp.mean(sqrt_v),
    sqrt_v_var=jnp.var(sqrt_v)
  )


def stats(state, b1=.9, b2=.999):
  return {k: wandb.Histogram(np_histogram=v) if k.endswith('hist') else v
          for k, v in _stats(state, b1, b2).items()}


def optimizer(name):
  try:
    return getattr(optax_edam, name)
  except AttributeError:
    return getattr(optax, name)
