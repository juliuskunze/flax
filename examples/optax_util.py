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


@partial(jit, static_argnames='momentum')
def _stats(state, momentum):
  opt_state = state.opt_state[0]
  if not hasattr(opt_state, 'mu'):
    return dict()

  grad_abs = _tree_flatten(tree_util.tree_map(
    lambda x: jnp.abs(bias_correction(x, momentum, opt_state.count + 1)),
    opt_state.mu))

  return dict(grad_abs_log_hist=jnp.histogram(jnp.nan_to_num(jnp.log(grad_abs), neginf=-88), bins=128),
              grad_abs_mean=jnp.mean(grad_abs),
              grad_abs_var=jnp.var(grad_abs))


def stats(state, momentum=.9):
  return {k: wandb.Histogram(np_histogram=v) if k.endswith('hist') else v
          for k, v in _stats(state, momentum).items()}


def optimizer(name):
  try:
    return getattr(optax_edam, name)
  except AttributeError:
    return getattr(optax, name)
