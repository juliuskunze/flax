import wandb

class DotDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

c = DotDict(optimizer='Ladam', learning_rate=1e-3, beta1=.9, dataset="cifar10", num_epochs=5, batch_size=128)

from absl import logging

import jax
from jax import random
import jax.numpy as jnp

jax.config.enable_omnistaging()

import ml_collections

import numpy as onp

import tensorflow_datasets as tfds
import tensorflow as tf

from flax import linen as nn, optim
from flax.metrics import tensorboard

class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x


def get_initial_params(images, key):
  return CNN().init(key, images)['params']


def create_optimizer(params):
  Optimizer = dict(Ladam=Ladam, Adam=optim.Adam)[c.optimizer]
  return Optimizer(c.learning_rate, beta1=c.beta1, beta2=c.beta2).create(params)


def onehot(labels, num_classes=10):
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels):
  return -jnp.mean(jnp.sum(onehot(labels) * logits, axis=-1))


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
    'loss': loss,
    'accuracy': accuracy,
  }
  return metrics


@jax.jit
def train_step(optimizer, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = CNN().apply({'params': params}, batch['image'])
    loss = cross_entropy_loss(logits, batch['label'])
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, batch['label'])
  return optimizer, metrics


@jax.jit
def eval_step(params, batch):
  logits = CNN().apply({'params': params}, batch['image'])
  return compute_metrics(logits, batch['label'])


def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm] for k, v in train_ds.items()}
    optimizer, metrics = train_step(optimizer, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
    k: onp.mean([metrics[k] for metrics in batch_metrics_np])
    for k in batch_metrics_np[0]}

  logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
               epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100)

  return optimizer, epoch_metrics_np


def eval_model(model, test_ds):
  metrics = eval_step(model, test_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_map(lambda x: x.item(), metrics)
  return summary['loss'], summary['accuracy']


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder(c.dataset)
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  if 'id' in train_ds: del train_ds['id']
  if 'id' in test_ds: del test_ds['id']
  return train_ds, test_ds

def train_and_evaluate():
  train_ds, test_ds = get_datasets()
  rng = random.PRNGKey(0)

  summary_writer = tensorboard.SummaryWriter(wandb.run.name)
  summary_writer.hparams(c)

  rng, init_rng = random.split(rng)
  params = get_initial_params(jnp.ones((1,) + train_ds['image'].shape[1:], jnp.float32), init_rng)
  optimizer = create_optimizer(params)

  for epoch in range(1, c.num_epochs + 1):
    rng, input_rng = random.split(rng)
    optimizer, train_metrics = train_epoch(
      optimizer, train_ds, c.batch_size, epoch, input_rng)
    loss, accuracy = eval_model(optimizer.target, test_ds)

    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                 epoch, loss, accuracy * 100)

    summary_writer.scalar('train_loss', train_metrics['loss'], epoch)
    summary_writer.scalar('train_accuracy', train_metrics['accuracy'], epoch)
    summary_writer.scalar('eval_loss', loss, epoch)
    summary_writer.scalar('eval_accuracy', accuracy, epoch)

  summary_writer.flush()
  return optimizer

def main():
  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  for dataset in ['cifar10']:
    c.dataset = dataset
    for optimizer in ['Adam']:
      c.optimizer = optimizer
      c.beta2 = .999 if optimizer == 'Adam' else .9
      for seed in [23]:
        c.seed = seed
        with wandb.init(entity='bandits', project='ladam', config=c, sync_tensorboard=True):
          train_and_evaluate()

if __name__ == '__main__':
  main()