import numpy as onp
import wandb
from jax import numpy as jnp

from flax import optim
from linen_examples.pixelcnn.introspection import apply_gradient_introspected, log


class DotDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


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
      o, metrics, histograms = apply_gradient_introspected(o, grad=make_dict(grads[step]))
      log(dict(step=step, **metrics), histograms=histograms)


if __name__ == '__main__':
  main(DotDict(learning_rate=.001, one_minus_beta1=.1, one_minus_beta2=.01, outlier=100000, experiment='one_outlier',
               outlier_step=2000, steps=5000, optimizer='Adam'))
