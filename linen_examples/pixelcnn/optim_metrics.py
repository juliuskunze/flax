from itertools import chain

import jax
import numpy as onp
import wandb
from jax import numpy as jnp
from jax.random import PRNGKey, choice

from flax import optim


class DotDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

def dict_union(*args):
  return dict(chain.from_iterable(d.items() for d in args))

def log(d, histograms):
  d = {k: (onp.array(v) if isinstance(v, jnp.DeviceArray) else v) for k, v in d.items()}
  histograms = {k: wandb.Histogram(np_histogram=[onp.array(v) if isinstance(v, jnp.DeviceArray) else v for v in v]) for k, v in histograms.items()}
  wandb.log(dict_union(d, histograms))

@jax.jit
def apply_gradient_observed(o, grad):
  new_o = o.apply_gradient(grad)
  tree = jax.tree_structure(grad)

  def concat(values): return jnp.concatenate([jnp.ravel(v) for v in values])
  def some(values): return choice(PRNGKey(0), values, (10,), replace=False)

  grad = concat(tree.flatten_up_to(grad))
  param = concat(tree.flatten_up_to(o.target))
  new_param = concat(tree.flatten_up_to(new_o.target))
  param_diff = param - new_param
  metrics = dict(grad=grad, param=param, param_diff=param_diff,
                 lr_factor=jnp.abs(param_diff) / o.optimizer_def.hyper_params.learning_rate)
  states = tree.flatten_up_to(o.state.param_states)
  new_states = tree.flatten_up_to(new_o.state.param_states)
  for field in states[0].__dataclass_fields__:
    old = concat(getattr(s, field) for s in states)
    new = concat(getattr(s, field) for s in new_states)
    metrics[str(field)] = new
    metrics[f'{field}_diff'] = new - old

  histograms = {name: jnp.histogram(value) for name, value in metrics.items()}
  metrics = dict_union(
    {name + '_norm': jnp.linalg.norm(m) for name, m in metrics.items()},
    *[{f'example/{i}/{name}': v for name, s in metrics.items() for i, v in enumerate(some(s))}])
  return new_o, metrics, histograms


def main(c):
  c.beta1 = 1 - c.one_minus_beta1
  c.beta2 = 1 - c.one_minus_beta2

  def effective_sample_size(beta): return (beta + 1) / (1 - beta)

  def make_dict(x): return dict(a=x, b=x * 10, c=x*50, d=x*1000)
  c.effective_sample_size1 = effective_sample_size(c.beta1)
  c.effective_sample_size2 = effective_sample_size(c.beta2)
  with wandb.init(config=c):
    grads = onp.random.normal(1, 1, size=(c.steps,))
    grads[c.outlier_step] = c.outlier
    o = getattr(optim, c.optimizer)(learning_rate=c.learning_rate, beta1=c.beta1, beta2=c.beta2).create(
      target=make_dict(jnp.zeros(())))
    for step in range(len(grads)):
      o, metrics, histograms = apply_gradient_observed(o, grad=make_dict(grads[step]))
      log(dict(step=step, **metrics), histograms=histograms)


if __name__ == '__main__':
  main(DotDict(learning_rate=.001, one_minus_beta1=.1, one_minus_beta2=.01, outlier=100000, experiment='one_outlier',
               outlier_step=2000, steps=5000, optimizer='Adam'))
