import argparse
import base64
import os
from pathlib import Path
from io import BytesIO
import time
import requests
from threading import Thread

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from consts import DEFAULT_IMG_OUTPUT_DIR
from utils import parse_arg_boolean, parse_arg_dalle_version
from consts import ModelSize

app = Flask(__name__)
CORS(app)
print("--> Starting Stable Diffusion img2img Server. This might take up to two minutes.")

from img2imgmodel import SDModel
img2img_model = None

curJobs = {
    'img2img': None
}

parser = argparse.ArgumentParser(description = "A Stable Diffusion img2img app to turn your textual prompts into visionary delights")
parser.add_argument("--port", type=int, default=8000, help = "backend port")
parser.add_argument("--save_to_disk", type = parse_arg_boolean, default = False, help = "Should save generated images to disk")
parser.add_argument("--img_format", type = str.lower, default = "JPEG", help = "Generated images format", choices=['jpeg', 'png'])
parser.add_argument("--output_dir", type = str, default = DEFAULT_IMG_OUTPUT_DIR, help = "Customer directory for generated images")
args, unknown = parser.parse_known_args()

def encodeImgs(generated_imgs):
    returned_generated_images = []    
    for idx, img in enumerate(generated_imgs):
        buffered = BytesIO()
        img.save(buffered, format=args.img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_str = f'data:image/{args.img_format};base64,{img_str}'
        returned_generated_images.append(img_str)
    return (returned_generated_images)

def postResponse(jobId, response):
    url = 'https://feedback-backend.herokuapp.com/create/callback'
    myobj = {}
    myobj[jobId] = response
    x = requests.post(url, data=myobj)
    return (x)

def doImg2Img():
    # Parse request
    job = curJobs['img2img']
    jobId = job['_id']
    jobData = job['data']
    image = jobData["image"]
    prompt = jobData["prompt"]
    num_images = jobData["num_images"]
    num_steps = jobData["num_steps"]

    # Generate Images
    generated_imgs = img2img_model.generate_images(BytesIO(base64.b64decode(image)), prompt, num_images, num_steps)
    curJobs['img2img'] = None

    # Encode Images
    returned_generated_images = encodeImgs(generated_imgs)
    
    # Return Images
    return postResponse(jobId, jsonify({
        'generatedImgs': returned_generated_images,
        'generatedImgsFormat': args.img_format
    }))

@app.route("/img2img", methods=["POST"])
@cross_origin()
def img2img_api():
    # Return if busy
    if (curJobs['img2img']): return jsonify(curJobs['img2img']['_id'], 'waiting')

    # Parse request
    job = request.get_json(force=True)

    # Run Job
    curJobs['img2img'] = job
    Thread(target = doImg2Img()).start()
    
    # Report Job has started
    return jsonify(job['_id'], 'working')


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
    # Initialize Models
    img2img_model = SDModel()

    # Run Warm-Up Tests
    # t1 = img2img_model.generate_images(BytesIO(base64.b64decode(tImg)), prompt, 1, 50)
    print("--> Stable Diffusion img2img Server is up and running!")


if __name__ == "__main__":
    app.run()
