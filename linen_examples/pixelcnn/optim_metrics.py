import numpy as np
import wandb
from jax import numpy as jnp

from flax import optim


class DotDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


def log(d):
  wandb.log({k: (np.array(v) if isinstance(v, jnp.DeviceArray) else v) for k, v in d.items()})


def main(c):
  c.beta1 = 1 - c.one_minus_beta1
  c.beta2 = 1 - c.one_minus_beta2

  def effective_sample_size(beta): return (beta + 1) / (1 - beta)

  c.effective_sample_size1 = effective_sample_size(c.beta1)
  c.effective_sample_size2 = effective_sample_size(c.beta2)
  with wandb.init(config=c):
    grads = np.random.normal(1, 1, size=(c.steps,))
    grads[c.outlier_step] = c.outlier
    o = getattr(optim, c.optimizer)(learning_rate=c.learning_rate, beta1=c.beta1, beta2=c.beta2).create(
      target=jnp.zeros(()))
    for step in range(len(grads)):
      grad = grads[step]
      new_o = o.apply_gradient(grad)
      param_diff = o.target - new_o.target

      d = dict(step=step, grad=grad, param=o.target, param_diff=param_diff, lr_factor=jnp.abs(param_diff) / c.learning_rate)

      for field in o.state.param_states.__dataclass_fields__:
        old = getattr(o.state.param_states, field)
        new = getattr(new_o.state.param_states, field)
        d[str(field)] = new
        d[f'{field}_diff'] = new - old

      o = new_o
      log(d)


if __name__ == '__main__':
  main(DotDict(learning_rate=.001, one_minus_beta1=.1, one_minus_beta2=.01, outlier=100000, experiment='one_outlier',
               outlier_step=2000, steps=25000, optimizer='Adam'))
