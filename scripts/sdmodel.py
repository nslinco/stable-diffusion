import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

# Speed it up, baby
from torch.backends.cuda import sdp_kernel
# torch.backends.cuda.enable_mem_efficient_sdp(True)
# torch.backends.cuda.sdp_kernel(False, False, True)
# torch.backends.cudnn.deterministic = True

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from diffusers import StableDiffusionPipeline

from slavehelper import postUpdate
import base64
from io import BytesIO
import boto3

client = boto3.client('s3', region_name='us-west-1')

# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     assert x_checked_image.shape[0] == len(has_nsfw_concept)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept


class SDModel:
    def __init__(self) -> None:
        self.optn_samples=1
        self.optddim_steps=50
        self.optplms=True
        self.optlaion400m=False
        self.optfixed_code=True
        self.optddim_eta=0.0
        self.optn_iter=1
        self.optH=512
        self.optW=512
        self.optC=4
        self.optf=8
        self.optscale=7.5
        self.optconfig="configs/stable-diffusion/v1-inference.yaml"
        self.optckpt="/var/meadowrun/machine_cache/sd-v1-4.ckpt"
        self.optseed=42
        self.optprecision="autocast"
        
        # self.frames = []

        # if self.optlaion400m:
        #     print("Falling back to LAION 400M model...")
        #     self.optconfig = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        #     self.optckpt = "models/ldm/text2img-large/model.ckpt"

        # seed_everything(self.optseed)

        # config = OmegaConf.load(f"{self.optconfig}")
        # self.model = load_model_from_config(config, f"{self.optckpt}")

        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.model = self.model.to(self.device)
        # self.model = torch.compile(self.model)

        self.model_id = "stabilityai/stable-diffusion-2-1-base"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, safety_checker=None)
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        # if self.optplms:
        #     self.sampler = PLMSSampler(self.model)
        # else:
        #     self.sampler = DDIMSampler(self.model)

        # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        # wm = "StableDiffusionV1"
        # self.wm_encoder = WatermarkEncoder()
        # self.wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
        print("Finished initializing the Stable Diffusion model")

    # def sampleCallback(self, img, i):
    #     # wtf is this voodoo magic?
    #     x_samples_ddim = self.model.decode_first_stage([img])
    #     x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    #     x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    #     x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    #     x_sample = 255. * rearrange(x_checked_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    #     newFrame = Image.fromarray(x_sample.astype(np.uint8))
    #     self.frames.append(newFrame)

    def reinitializeModel(self, newConfig, newCheckpoint):
        # Update Vars
        self.optconfig = newConfig
        self.optckpt = newCheckpoint

        # Re-Initialize
        config = OmegaConf.load(f"{self.optconfig}")
        self.model = load_model_from_config(config, f"{self.optckpt}")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

    def generate_images(
        self,
        prompt: str,
        optn_samples=1,
        optddim_steps=50,
        optplms=True,
        optfixed_code=True,
        optddim_eta=0.0,
        optn_iter=1,
        optH=512,
        optW=512,
        optC=4,
        optf=8,
        optscale=7.5,
        optprecision="autocast",
        optseed=42
    ):
        print(f"Generating {optn_samples} images from prompt: {prompt}")
        
        tic = time.time()

        # Seed Everything
        seed_everything(optseed)

        batch_size = optn_samples

        assert prompt is not None
        data = [batch_size * [prompt]]

        if (not optplms):
            self.sampler = DDIMSampler(self.model)

        start_code = None
        if optfixed_code:
            start_code = torch.randn([optn_samples, optC, optH // optf, optW // optf], device=self.device)

        outimgs = []

        precision_scope = autocast if optprecision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for n in trange(optn_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if optscale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [optC, optH // optf, optW // optf]
                            samples_ddim, _ = self.sampler.sample(S=optddim_steps,
                                                            conditioning=c,
                                                            batch_size=batch_size,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=optscale,
                                                            unconditional_conditioning=uc,
                                                            eta=optddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                            # x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, self.wm_encoder)
                                outimgs.append(img)
                    toc = time.time()
        print(f"Generated {len(outimgs)} images from prompt: [{prompt}] in {toc-tic}s")
        return(outimgs)

    def generate_images_bulk(
        self,
        instanceIp,
        jobId,
        subJobs,
        optddim_steps=50,
        animate=False,
        mask=None
    ):
        optn_samples=1
        optplms=False
        optfixed_code=True
        optn_iter=1
        optH=512
        optW=512
        optC=4
        optf=8
        optprecision="autocast"
        logRate = 1 if animate else 100

        numerator = 0
        denominator = optn_samples * len(subJobs)
        print(f"Generating {denominator} images for bulk job: {jobId}")
        
        tic = time.time()

        batch_size = optn_samples

        if (not optplms):
            self.sampler = DDIMSampler(self.model)

        start_code = None
        if optfixed_code:
            start_code = torch.randn([optn_samples, optC, optH // optf, optW // optf], device=self.device)

        outimgs = []
        optseed = None
        optddim_eta = None
        optscale = None

        precision_scope = autocast if optprecision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for n in trange(optn_iter, desc="Sampling"):
                        for subJob in subJobs:
                            # Start the clock
                            iTic = time.time()

                            # Parse job
                            optjobUID = subJob['jobUID']
                            optprompt = subJob['prompt']
                            optmodifier = subJob['modifier']

                            prompt = optprompt + ', ' + optmodifier

                            data = [batch_size * [prompt]]

                            if (optseed != subJob['seed']):
                                optseed = subJob['seed']
                                seed_everything(optseed)
                            optddim_eta = subJob['eta']
                            optscale = subJob['scale']

                            # Iterate why?
                            for prompts in tqdm(data, desc="data"):
                                uc = None
                                if optscale != 1.0:
                                    uc = self.model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = self.model.get_learned_conditioning(prompts)
                                shape = [optC, optH // optf, optW // optf]
                                samples_ddim, intermediates = self.sampler.sample(S=optddim_steps,
                                                                conditioning=c,
                                                                batch_size=batch_size,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=optscale,
                                                                unconditional_conditioning=uc,
                                                                eta=optddim_eta,
                                                                x_T=start_code,
                                                                log_every_t=logRate,
                                                                mask=mask)

                                # Initialize Animation fn
                                gifName = None
                                if (animate):
                                    # Break down output
                                    intermediates = intermediates['pred_x0']
                                    
                                    # Decode Intermediates
                                    print(f'intermediates: {len(intermediates)}')
                                    for intermediate in intermediates:
                                        x_intermediates_ddim = self.model.decode_first_stage(intermediate)
                                        x_intermediates_ddim = torch.clamp((x_intermediates_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                        x_intermediates_ddim = x_intermediates_ddim.cpu().permute(0, 2, 3, 1).numpy()
                                        x_intermediates_torch = torch.from_numpy(x_intermediates_ddim).permute(0, 3, 1, 2)
                                        for x_sample in x_intermediates_torch:
                                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                            img = Image.fromarray(x_sample.astype(np.uint8))
                                            self.frames.append(img)
                                    self.frames.pop(0)
                                    print(f'frames: {len(self.frames)}')

                                    # Add reverse frames for bounce effect
                                    reversedFrames = self.frames[::-1]
                                    reversedFrames.extend(self.frames)
                                    print(f'extended frames: {len(reversedFrames)}')
                                    
                                    # Create animation
                                    buffered = BytesIO()
                                    gif = reversedFrames[0]
                                    gif.save(
                                        fp=buffered,
                                        format='GIF',
                                        append_images=reversedFrames,
                                        save_all=True,
                                        duration=(int)(1000/len(reversedFrames)),
                                        loop=0
                                    )
                                    # gif_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                    # gif_str = f'data:image/gif;base64,{gif_str}'
                                    self.frames = []

                                    # Name GIF
                                    gifName = f"{jobId}-{optjobUID}.gif"

                                    # Upload to s3 bucket
                                    buffered.seek(0)
                                    uploaded = client.upload_fileobj(
                                        buffered,
                                        'meadowrun-sd-69',
                                        'animations/{}'.format(gifName),
                                        ExtraArgs={'ACL':'public-read'}
                                    )
                                    print(f"Uploaded to S3 bucket: {uploaded}")

                                # Decode Samples
                                x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                                # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                                # x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                                x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                                # Create result
                                for x_sample in x_checked_image_torch:
                                    # Format Image
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    buffered = BytesIO()
                                    img.save(fp=buffered, format='jpeg')

                                    # Name image
                                    imgName = f"{jobId}-{optjobUID}.jpeg"

                                    # Upload to s3 bucket
                                    buffered.seek(0)
                                    client.upload_fileobj(
                                        buffered,
                                        'meadowrun-sd-69',
                                        'images/{}'.format(imgName),
                                        ExtraArgs={'ACL':'public-read'}
                                    )

                                    # img = put_watermark(img, self.wm_encoder)
                                    iToc = time.time()
                                    iTime = iToc-iTic
                                    retObj = {
                                        "result": imgName,
                                        "params": {
                                            "prompt": optprompt,
                                            "modifier": optmodifier,
                                            "seed": optseed,
                                            "eta": optddim_eta,
                                            "scale": optscale,
                                        },
                                        "time": iTime,
                                        "animation": gifName
                                    }
                                    outimgs.append(retObj)
                                    numerator += 1
                                    postUpdate(jobId, {
                                        "numerator": numerator,
                                        "denominator": denominator,
                                        "instanceIp": instanceIp
                                    })
                                    print(f"Generated image {numerator}/{denominator} in {iTime}s")
        toc = time.time()
        print(f"Generated {len(outimgs)} images from bulk prompt: [{prompt}] in {toc-tic}s")
        return(outimgs)

    def generate_images_quick(
        self,
        job
    ):
        # Parse job
        jobId = job["parentId"]
        optjobUID = job["jobUID"]
        animate = job["animate"]
        optddim_steps = job["ddim_steps"]
        optprompt = job["prompt"]
        optmodifier = job["modifier"]
        optseed = job["seed"]
        optddim_eta = job["eta"]
        optscale = job["scale"]

        prompt = optprompt + ', ' + optmodifier

        optn_samples=1
        optplms=False
        optfixed_code=True
        optn_iter=1
        optH=512
        optW=512
        optC=4
        optf=8
        optprecision="autocast"
        logRate = 1 if animate else 100

        print(f"Generating image for bulk job: {jobId}-{optjobUID}")
        
        tic = time.time()

        seed_everything(optseed)

        batch_size = optn_samples
        data = [batch_size * [prompt]]

        if (not optplms):
            self.sampler = DDIMSampler(self.model)

        start_code = None
        if optfixed_code:
            start_code = torch.randn([optn_samples, optC, optH // optf, optW // optf], device=self.device)

        outimgs = []

        precision_scope = autocast if optprecision=="autocast" else nullcontext
        # with sdp_kernel({"enable_math": False, "enable_flash": False, "enable_mem_efficient": True}):
        with sdp_kernel(False, True):
            with torch.no_grad():
                with precision_scope("cuda"):
                    with self.model.ema_scope():
                        for n in trange(optn_iter, desc="Sampling"):
                            # Start the clock
                            iTic = time.time()

                            # Iterate why?
                            for prompts in tqdm(data, desc="data"):
                                uc = None
                                if optscale != 1.0:
                                    uc = self.model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = self.model.get_learned_conditioning(prompts)
                                shape = [optC, optH // optf, optW // optf]
                                samples_ddim, intermediates = self.sampler.sample(S=optddim_steps,
                                                                conditioning=c,
                                                                batch_size=batch_size,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=optscale,
                                                                unconditional_conditioning=uc,
                                                                eta=optddim_eta,
                                                                x_T=start_code,
                                                                log_every_t=logRate)

                                # Initialize Animation fn
                                gifName = None
                                if (animate):
                                    # Break down output
                                    intermediates = intermediates['pred_x0']
                                    
                                    # Decode Intermediates
                                    # print(f'intermediates: {len(intermediates)}')
                                    for intermediate in intermediates:
                                        x_intermediates_ddim = self.model.decode_first_stage(intermediate)
                                        x_intermediates_ddim = torch.clamp((x_intermediates_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                        x_intermediates_ddim = x_intermediates_ddim.cpu().permute(0, 2, 3, 1).numpy()
                                        x_intermediates_torch = torch.from_numpy(x_intermediates_ddim).permute(0, 3, 1, 2)
                                        for x_sample in x_intermediates_torch:
                                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                            img = Image.fromarray(x_sample.astype(np.uint8))
                                            self.frames.append(img)
                                    self.frames.pop(0)
                                    # print(f'frames: {len(self.frames)}')

                                    # Add reverse frames for bounce effect
                                    reversedFrames = self.frames[::-1]
                                    reversedFrames.extend(self.frames)
                                    # print(f'extended frames: {len(reversedFrames)}')
                                    
                                    # Create animation
                                    buffered = BytesIO()
                                    gif = reversedFrames[0]
                                    gif.save(
                                        fp=buffered,
                                        format='GIF',
                                        append_images=reversedFrames,
                                        save_all=True,
                                        duration=(int)(1000/len(reversedFrames)),
                                        loop=0
                                    )
                                    # gif_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                    # gif_str = f'data:image/gif;base64,{gif_str}'
                                    self.frames = []

                                    # Name GIF
                                    gifName = f"{jobId}-{optjobUID}.gif"

                                    # Upload to s3 bucket
                                    buffered.seek(0)
                                    uploaded = client.upload_fileobj(
                                        buffered,
                                        'meadowrun-sd-69',
                                        'animations/{}'.format(gifName),
                                        ExtraArgs={'ACL':'public-read'}
                                    )
                                    print(f"Uploaded to S3 bucket: {uploaded}")

                                # Decode Samples
                                x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                                # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                                # x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                                x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                                # Create result
                                for x_sample in x_checked_image_torch:
                                    # Format Image
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    buffered = BytesIO()
                                    img.save(fp=buffered, format='jpeg')

                                    # Name image
                                    imgName = f"{jobId}-{optjobUID}.jpeg"

                                    # Upload to s3 bucket
                                    buffered.seek(0)
                                    client.upload_fileobj(
                                        buffered,
                                        'meadowrun-sd-69',
                                        'images/{}'.format(imgName),
                                        ExtraArgs={'ACL':'public-read'}
                                    )

                                    # img = put_watermark(img, self.wm_encoder)
                                    iToc = time.time()
                                    iTime = iToc-iTic
                                    retObj = {
                                        "jobId": jobId,
                                        "jobUID": optjobUID,
                                        "result": imgName,
                                        "params": {
                                            "prompt": optprompt,
                                            "modifier": optmodifier,
                                            "seed": optseed,
                                            "eta": optddim_eta,
                                            "scale": optscale,
                                            "ddim_steps": optddim_steps
                                        },
                                        "time": iTime,
                                        "animation": gifName
                                    }
                                    outimgs.append(retObj)
        toc = time.time()
        print(f"Generated {len(outimgs)} images for bulk job: {jobId}-{optjobUID} in {toc-tic}s")
        return(outimgs)

    def generate_images_quicker(
        self,
        job
    ):
        # Parse job
        jobId = job["parentId"]
        optjobUID = job["jobUID"]
        animate = job["animate"]
        optddim_steps = job["ddim_steps"]
        optprompt = job["prompt"]
        optmodifier = job["modifier"]
        optseed = job["seed"]
        optddim_eta = job["eta"]
        optscale = job["scale"]

        prompt = optprompt + ', ' + optmodifier

        optn_samples=1
        optplms=False
        optfixed_code=True
        optn_iter=1
        optH=512
        optW=512
        optC=4
        optf=8
        optprecision="autocast"
        logRate = 1 if animate else 100

        print(f"Generating image for bulk job: {jobId}-{optjobUID}")

        # Name image
        gifName = f"{jobId}-{optjobUID}.gif"
        imgName = f"{jobId}-{optjobUID}.jpeg"
        
        tic = time.time()

        uTic = 0
        uToc = 0
        uTime = 0

        with torch.autocast("cuda"):
            image = self.pipe(
                prompt=prompt,
                height=optH,
                width=optW,
                num_inference_steps=optddim_steps,
                guidance_scale=optscale,
                eta=optddim_eta,
                num_images_per_prompt=optn_samples,
                output_type='pil',
                generator = torch.Generator(device="cuda").manual_seed(optseed)
            ).images[0]

            # Upload to s3 bucket
            # img = Image.fromarray(image.astype(np.uint8)) # Faster to leave output_type as default?
            uTic = time.time()
            buffered = BytesIO()
            image.save(fp=buffered, format='jpeg')
            buffered.seek(0)
            client.upload_fileobj(
                buffered,
                'meadowrun-sd-69',
                'images/{}'.format(imgName),
                ExtraArgs={'ACL':'public-read'}
            )
            uToc = time.time()
            uTime = uToc-uTic

        outimgs = []

        toc = time.time()
        iTime = toc-tic

        retObj = {
            "jobId": jobId,
            "jobUID": optjobUID,
            "modelId": self.model_id,
            "result": imgName,
            "params": {
                "prompt": optprompt,
                "modifier": optmodifier,
                "seed": optseed,
                "eta": optddim_eta,
                "scale": optscale,
                "ddim_steps": optddim_steps
            },
            "time": iTime,
            "animation": gifName
        }
        outimgs.append(retObj)

        print(f"Generated {len(outimgs)} images for bulk job: {jobId}-{optjobUID} in {iTime-uTime}-{uTime}s")
        return(outimgs)