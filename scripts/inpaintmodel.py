import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


class InpaintModel:
    def __init__(self) -> None:
        self.optindir=''                            # dir containing image-mask pairs (`example.png` and `example_mask.png`)",
        self.optoutdir='outputs/inpaint-samples'    # dir to write results to

        config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(torch.load("/var/meadowrun/machine_cache/last.ckpt")["state_dict"],
                            strict=False)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.sampler = DDIMSampler(self.model)


    def generate_image(self, image, mask, num_steps: int):
        # os.makedirs(self.optoutdir, exist_ok=True)
        with torch.no_grad():
            with self.model.ema_scope():
                # outpath = os.path.join(self.optoutdir, os.path.split(image)[1])
                batch = make_batch(image, mask, device=self.device)

                # encode masked image and concat downsampled mask
                c = self.model.cond_stage_model.encode(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                    size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1]-1,)+c.shape[2:]
                samples_ddim, _ = self.sampler.sample(S=num_steps,
                                                conditioning=c,
                                                batch_size=c.shape[0],
                                                shape=shape,
                                                verbose=False)
                x_samples_ddim = self.model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                            min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                img = Image.fromarray(inpainted.astype(np.uint8))
                return(img)
