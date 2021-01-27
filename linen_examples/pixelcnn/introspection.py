import jax
import numpy as onp
import wandb
from jax import numpy as jnp
from jax.random import choice, PRNGKey

@jax.jit
def apply_gradient_introspected(o, grad, learning_rate):
  if learning_rate is None:
    learning_rate = o.optimizer_def.hyper_params.learning_rate
  new_o = o.apply_gradient(grad, learning_rate=learning_rate)
  tree = jax.tree_structure(grad)

  def concat(values): return jnp.concatenate([jnp.ravel(v) for v in values])
  def some(values): return choice(PRNGKey(0), values, (10,), replace=False)

  grad = concat(tree.flatten_up_to(grad))
  param = concat(tree.flatten_up_to(o.target))
  new_param = concat(tree.flatten_up_to(new_o.target))
  param_diff = param - new_param
  metrics = dict(grad=grad, param=param, param_diff=param_diff, lr_factor=jnp.abs(param_diff) / learning_rate)
  states = tree.flatten_up_to(o.state.param_states)
  new_states = tree.flatten_up_to(new_o.state.param_states)
  for field in states[0].__dataclass_fields__:
    old = concat(getattr(s, field) for s in states)
    new = concat(getattr(s, field) for s in new_states)
    metrics[str(field)] = new
    metrics[f'{field}_diff'] = new - old

  histograms = {name: jnp.histogram(value) for name, value in metrics.items()}
  metrics = dict(
    **{name + '_norm': jnp.linalg.norm(m) for name, m in metrics.items()},
    **{f'example/{i}/{name}': v for name, s in metrics.items() for i, v in enumerate(some(s))})
  return new_o, metrics, histograms


def log(d, histograms):
  d = {k: (onp.array(v) if isinstance(v, jnp.DeviceArray) else v) for k, v in d.items()}
  histograms = {k: wandb.Histogram(np_histogram=[onp.array(v) if isinstance(v, jnp.DeviceArray) else v for v in v]) for k, v in histograms.items()}
  d.update(histograms)
  wandb.log(d)