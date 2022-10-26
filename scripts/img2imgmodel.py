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
        # Initialize Variables

        # Unused
        # self.optinit_img='' # "path to the input image"
        # self.optfixed_code=True # "if enabled, uses the same starting code across all samples ",
        # self.optf=8 # "downsampling factor, most often 8 or 16",
        # self.optC=4 # "latent channels",
        # self.optn_rows=0 # "rows in the grid (default: n_samples)",
        # self.optn_iter=1 # "sample this often",
        # self.optskip_grid=True # "do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        # self.optskip_save=True # "do not save indiviual samples. For speed measurements.",
        # self.optoutdir="outputs/img2img-samples" # "dir to write results to",

        # Global Variables
        self.optseed=42 # "the seed (for reproducible sampling)",
        self.optconfig="configs/stable-diffusion/v1-inference.yaml" # "path to config which constructs model",
        self.optckpt="/var/meadowrun/machine_cache/sd-v1-4.ckpt" # "path to checkpoint of model",
        self.optplms=False # "use plms sampling",

        # Generation-Local Variables
        self.optddim_steps=50 # "number of ddim sampling steps",
        self.optddim_eta=0.0 # "ddim eta (eta=0.0 corresponds to deterministic sampling",
        self.optn_samples=1 # "how many samples to produce for each given prompt. A.k.a batch size",
        self.optscale=5.0 # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        self.optstrength=0.75 # "strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
        self.optprecision="autocast"


        # Seed eeeverything?
        seed_everything(self.optseed)

        # Configure Model
        config = OmegaConf.load(f"{self.optconfig}")
        self.model = load_model_from_config(config, f"{self.optckpt}")

        # GFX Card Setup
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        # Sampling Setup
        if self.optplms:
            # raise NotImplementedError("PLMS sampler not (yet) supported")
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)


    def generate_images(
            self,
            image,
            prompt: str,
            optplms=False,
            optn_samples=1,
            optddim_steps=50,
            optddim_eta=0.0,
            optscale=5.0,
            optstrength=0.75,
            optprecision="autocast",
            optseed=42
    ):

        # Start the clock
        tic = time.time()

        # Seed Everything
        seed_everything(optseed)

        # Initialize the batch size
        batch_size = optn_samples if optn_samples else optn_samples

        # Initialize the prompts
        assert prompt is not None
        data = [batch_size * [prompt]]

        # Change sampler
        if (optplms):
            self.sampler = PLMSSampler(self.model)

        # Load starting image
        init_image = load_img(image).to(self.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

        # Set up sampler?
        self.sampler.make_schedule(ddim_num_steps=optddim_steps, ddim_eta=optddim_eta, verbose=False)

        # Scale optddim_steps by strength
        assert 0. <= optstrength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(optstrength * optddim_steps)
        print(f"target t_enc is {t_enc} steps")

        # Initialize return array
        outimgs = []

        # Start the fun stuff
        precision_scope = autocast if optprecision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    # Iterate over each generation
                    for n in trange(num_images, desc="Sampling"):
                        # Iterate over each prompt
                        for prompts in tqdm(data, desc="data"):
                            # ?
                            uc = None
                            if optscale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            
                            # Tokenize prompt?
                            c = self.model.get_learned_conditioning(prompts)

                            # Cncode (scaled latent)
                            z_enc = self.sampler.stochastic_encode(
                                init_latent,
                                torch.tensor([t_enc]*batch_size).to(self.device)
                            )
                            
                            # Decode it
                            samples = self.sampler.decode(
                                z_enc,
                                c,
                                t_enc,
                                unconditional_guidance_scale=optscale,
                                unconditional_conditioning=uc,
                            )

                            # ?
                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            
                            # Format Images
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                outimgs.append(img)

                            # Save Result
                            all_samples.append(x_samples)

                    # Stop the clock
                    toc = time.time()

        # Log Finished Job
        print(f"Generated {len(outimgs)} images from prompt: [{prompt}] in {toc-tic}s")

        # Return the results
        return(outimgs)
