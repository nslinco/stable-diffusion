import argparse
import base64
import os
from pathlib import Path
from io import BytesIO
import time

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from consts import DEFAULT_IMG_OUTPUT_DIR
from utils import parse_arg_boolean, parse_arg_dalle_version
from consts import ModelSize

app = Flask(__name__)
CORS(app)
print("--> Starting Stable Diffusion img2img Server. This might take up to two minutes.")

from img2imgmodel import SDModel
sd_model = None

parser = argparse.ArgumentParser(description = "A Stable Diffusion img2img app to turn your textual prompts into visionary delights")
parser.add_argument("--port", type=int, default=8000, help = "backend port")
parser.add_argument("--save_to_disk", type = parse_arg_boolean, default = False, help = "Should save generated images to disk")
parser.add_argument("--img_format", type = str.lower, default = "JPEG", help = "Generated images format", choices=['jpeg', 'png'])
parser.add_argument("--output_dir", type = str, default = DEFAULT_IMG_OUTPUT_DIR, help = "Customer directory for generated images")
args, unknown = parser.parse_known_args()

@app.route("/img2img", methods=["POST"])
@cross_origin()
def generate_images_api():
    json_data = request.get_json(force=True)
    image = json_data["image"]
    prompt = json_data["prompt"]
    num_steps = json_data["num_steps"]
    generated_imgs = sd_model.generate_image(BytesIO(base64.b64decode(image)), prompt, num_steps)

    returned_generated_images = []
    # if args.save_to_disk: 
    #     dir_name = os.path.join(args.output_dir,f"{time.strftime('%Y-%m-%d_%H:%M:%S')}_{text_prompt}")
    #     Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # for idx, img in enumerate(generated_img):
        # if args.save_to_disk: 
        #   img.save(os.path.join(dir_name, f'{idx}.{args.img_format}'), format=args.img_format)

    for idx, img in enumerate(generated_imgs):
        # if args.save_to_disk: 
        #   img.save(os.path.join(dir_name, f'{idx}.{args.img_format}'), format=args.img_format)

        buffered = BytesIO()
        img.save(buffered, format=args.img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        returned_generated_images.append(img_str)

    # print(f"Created {num_images} images from text prompt [{text_prompt}]")
    
    response = {'generatedImgs': returned_generated_images,
    'generatedImgsFormat': args.img_format}
    return jsonify(response)


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
    sd_model = SDModel()
    #ip_model.generate_image("warmup.png", "warmup_mask.png", 50)
    print("--> Stable Diffusion img2img Server is up and running!")


if __name__ == "__main__":
    app.run()
