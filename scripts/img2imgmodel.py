"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


class SDModel:
    def __init__(self) -> None:
        self.optinit_img='' # "path to the input image"
        self.optoutdir="outputs/img2img-samples" # "dir to write results to",
        self.optskip_grid=True # "do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        self.optskip_save=True # "do not save indiviual samples. For speed measurements.",
        self.optddim_steps=50 # "number of ddim sampling steps",
        self.optplms=False # "use plms sampling",
        self.optfixed_code=True # "if enabled, uses the same starting code across all samples ",
        self.optddim_eta=0.0 # "ddim eta (eta=0.0 corresponds to deterministic sampling",
        self.optn_iter=1 # "sample this often",
        self.optC=4 # "latent channels",
        self.optf=8 # "downsampling factor, most often 8 or 16",
        self.optn_samples=1 # "how many samples to produce for each given prompt. A.k.a batch size",
        self.optn_rows=0 # "rows in the grid (default: n_samples)",
        self.optscale=5.0 # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        self.optstrength=0.75 # "strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
        self.optconfig="configs/stable-diffusion/v1-inference.yaml" # "path to config which constructs model",
        self.optckpt="/var/meadowrun/machine_cache/sd-v1-4.ckpt" # "path to checkpoint of model",
        self.optseed=42 # "the seed (for reproducible sampling)",
        self.optprecision="autocast"

        seed_everything(self.optseed)

        config = OmegaConf.load(f"{self.optconfig}")
        self.model = load_model_from_config(config, f"{self.optckpt}")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        if self.optplms:
            raise NotImplementedError("PLMS sampler not (yet) supported")
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)

        # os.makedirs(self.optoutdir, exist_ok=True)
        # self.outpath = self.optoutdir


    def generate_images(self, image, prompt: str, num_images: int, num_steps: int):
        batch_size = self.optn_samples
        n_rows = self.optn_rows if self.optn_rows > 0 else batch_size

        assert prompt is not None
        data = [batch_size * [prompt]]

        # sample_path = os.path.join(self.outpath, "samples")
        # os.makedirs(sample_path, exist_ok=True)
        # base_count = len(os.listdir(sample_path))
        # grid_count = len(os.listdir(self.outpath)) - 1

        init_image = load_img(image).to(self.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

        self.sampler.make_schedule(ddim_num_steps=num_steps, ddim_eta=self.optddim_eta, verbose=False)

        assert 0. <= self.optstrength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(self.optstrength * num_steps)
        print(f"target t_enc is {t_enc} steps")

        outimgs = []

        precision_scope = autocast if self.optprecision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(num_images, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if self.optscale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self.device))
                            # decode it
                            samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=self.optscale,
                                                    unconditional_conditioning=uc,)

                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            if not self.optskip_save:
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(sample_path, f"{base_count:05}.png"))
                                    base_count += 1
                            
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                outimgs.append(img)

                            all_samples.append(x_samples)

                    if not self.optskip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(self.outpath, f'grid-{grid_count:04}.png'))
                        grid_count += 1

                    toc = time.time()
        print(f"Generated {len(outimgs)} images from prompt: [{prompt}] in {toc-tic}s")
        return(outimgs)
