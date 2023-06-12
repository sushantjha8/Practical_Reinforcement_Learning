import gym
import numpy as np
import cv2
import random
import torch
from tgan.vqgan import VQGAN
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from tgan.discriminator import Discriminator
from tgan.lpips import LPIPS
from tgan.uitils import load_data
from tgan.modelutils import weights_init

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
IMAGE_SIZE = 210
BATCH_SIZE = 2


class Custom_ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(Custom_ObservationWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low), self.observation(old_space.high), dtype=np.float32
        )

    def observation(self, observation):
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


def itrate_batches(envs, batch_size=BATCH_SIZE):
    # we will reset all env add to batch list
    for e in envs:
        e.reset()
    batch = []  # len of batch == len envs
    print(f" len of batch {len(batch)}")
    env_gen = iter(lambda: random.choice(envs), None)
    while True:
        e = next(env_gen)

        obs, reward, is_done, _, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
            print(f" this is mean_obs {np.mean(obs)}")
        if len(batch) == batch_size:
            bnp = np.array(batch, dtype=np.float32)
            bnp *= 2.0 / 255.0 - 1.0
            yield torch.tensor(bnp)
            batch.clear()
        if is_done:
            e.reset()


device = torch.device("cpu")

envs = [Custom_ObservationWrapper(gym.make(name)) for name in ("ALE/Breakout-v5", "AirRaid-v4")]


parser = argparse.ArgumentParser(description="VQGAN")
parser.add_argument(
    "--latent-dim",
    type=int,
    default=210,
    help="Latent dimension n_z (default: 256)",
)
parser.add_argument(
    "--image-size",
    type=int,
    default=210,
    help="Image height and width (default: 256)",
)
parser.add_argument(
    "--num-codebook-vectors",
    type=int,
    default=1024,
    help="Number of codebook vectors (default: 256)",
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.25,
    help="Commitment loss scalar (default: 0.25)",
)
parser.add_argument(
    "--image-channels",
    type=int,
    default=3,
    help="Number of channels of images (default: 3)",
)
parser.add_argument(
    "--dataset-path",
    type=str,
    default="/data",
    help="Path to data (default: /data)",
)
parser.add_argument("--device", type=str, default="cpu", help="Which device the training is on")
parser.add_argument(
    "--batch-size",
    type=int,
    default=6,
    help="Input batch size for training (default: 6)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to train (default: 50)",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=2.25e-05,
    help="Learning rate (default: 0.0002)",
)
parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta param (default: 0.0)")
parser.add_argument("--beta2", type=float, default=0.9, help="Adam beta param (default: 0.999)")
parser.add_argument(
    "--disc-start",
    type=int,
    default=10000,
    help="When to start the discriminator (default: 0)",
)
parser.add_argument("--disc-factor", type=float, default=1.0, help="")
parser.add_argument(
    "--rec-loss-factor",
    type=float,
    default=1.0,
    help="Weighting factor for reconstruction loss.",
)
parser.add_argument(
    "--perceptual-loss-factor",
    type=float,
    default=1.0,
    help="Weighting factor for perceptual loss.",
)

args = parser.parse_args()

vqgan = VQGAN(args)
for b in itrate_batches(envs):
    imgs = b.to(device)

    codebook_mapping, codebook_indices, q_loss = vqgan.encode(imgs)
    print(q_loss)
