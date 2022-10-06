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
print("--> Starting Stable Diffusion Server. This might take up to two minutes.")

from sdmodel import SDModel
# from inpaintmodel import InpaintModel
from img2imgmodel import SDModel as Img2ImgModel

curJobs = {
    'sd': None,
    'inpaint': None,
    'img2img': None
}

sd_model = None
# inpaint_model = None
img2img_model = None

parser = argparse.ArgumentParser(description = "A Stable Diffusion app to turn your textual prompts into visionary delights")
parser.add_argument("--port", type=int, default=8000, help = "backend port")
parser.add_argument("--model_version", type = parse_arg_dalle_version, default = ModelSize.MINI, help = "Mini, Mega, or Mega_full")
parser.add_argument("--save_to_disk", type = parse_arg_boolean, default = False, help = "Should save generated images to disk")
parser.add_argument("--img_format", type = str.lower, default = "JPEG", help = "Generated images format", choices=['jpeg', 'png'])
parser.add_argument("--output_dir", type = str, default = DEFAULT_IMG_OUTPUT_DIR, help = "Customer directory for generated images")
args = parser.parse_args()

def encodeImgs(generated_imgs):
    returned_generated_images = []    
    for idx, img in enumerate(generated_imgs):
        buffered = BytesIO()
        img.save(buffered, format=args.img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_str = f'data:image/${args.img_format};base64,${img_str}'
        returned_generated_images.append(img_str)
    return (returned_generated_images)

def postResponse(jobId, response):
    url = 'https://feedback-backend.herokuapp.com/create/callback'
    myobj = {}
    myobj[jobId] = response
    x = requests.post(url, json = myobj)
    return (x)

def doSD():
    # Parse request
    job = curJobs['sd']
    jobId = job._id
    jobData = job.data
    text_prompt = jobData["prompt"]
    num_images = jobData["num_images"]
    num_steps = jobData["num_steps"]

    # Generate Images
    generated_imgs = sd_model.generate_images(text_prompt, num_images, num_steps)
    curJobs['sd'] = None

    # Encode Images
    returned_generated_images = encodeImgs(generated_imgs)
    
    # Return Images
    return postResponse(jobId, jsonify({
        'generatedImgs': returned_generated_images,
        'generatedImgsFormat': args.img_format
    }))

def doInpaint():
    # Parse request
    job = curJobs['inpaint']
    jobId = job._id
    jobData = job.data
    image = jobData["image"]
    mask = jobData["mask"]
    num_steps = jobData["num_steps"]

    # Generate Images
    generated_img = inpaint_model.generate_image(BytesIO(base64.b64decode(image)), BytesIO(base64.b64decode(mask)), num_steps)
    curJobs['inpaint'] = None

    # Encode Images
    returned_generated_images = encodeImgs([generated_img])
    
    # Return Images
    return postResponse(jobId, jsonify({
        'generatedImgs': returned_generated_images,
        'generatedImgsFormat': args.img_format
    }))

def doImg2Img():
    # Parse request
    job = curJobs['img2img']
    jobId = job._id
    jobData = job.data
    image = jobData["image"]
    prompt = jobData["prompt"]
    num_images = jobData["num_images"]
    num_steps = jobData["num_steps"]

    # Generate Images
    generated_imgs = img2img_model.generate_image(BytesIO(base64.b64decode(image)), prompt, num_images, num_steps)
    curJobs['img2img'] = None

    # Encode Images
    returned_generated_images = encodeImgs(generated_imgs)
    
    # Return Images
    return postResponse(jobId, jsonify({
        'generatedImgs': returned_generated_images,
        'generatedImgsFormat': args.img_format
    }))

@app.route("/sd", methods=["POST"])
@cross_origin()
def sd_api():
    # Return if busy
    if (curJobs['sd']): return jsonify(jobId, 'waiting')

    # Parse request
    job = request.get_json(force=True)

    # Run Job
    curJobs['sd'] = job
    Thread(target = doSD()).start()
    
    # Report Job has started
    return jsonify(jobId, 'working')

@app.route("/inpaint", methods=["POST"])
@cross_origin()
def inpaint_api():
    # Return if busy
    if (curJobs['inpaint']): return jsonify(jobId, 'waiting')

    # Parse request
    job = request.get_json(force=True)

    # Run Job
    curJobs['inpaint'] = job
    Thread(target = doInpaint()).start()
    
    # Report Job has started
    return jsonify(jobId, 'working')

@app.route("/img2img", methods=["POST"])
@cross_origin()
def img2img_api():
    # Return if busy
    if (curJobs['img2img']): return jsonify(jobId, 'waiting')

    # Parse request
    job = request.get_json(force=True)

    # Run Job
    curJobs['img2img'] = job
    Thread(target = doImg2Img()).start()
    
    # Report Job has started
    return jsonify(jobId, 'working')


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
    # Initialize Models
    sd_model = SDModel()
    # inpaint_model = InpaintModel()
    img2img_model = Img2ImgModel()

    # Run Warm-Up Tests
    t1 = sd_model.generate_images("warm-up", 1, 50)
    print('t1: ', t1)
    print("--> Stable Diffusion Server is up and running!")
    # inpaint_model.generate_image(t1[0], t1[0], 50)
    # print("--> Inpainting Server is up and running!")
    img2img_model.generate_images(t1[0].split(',')[1], "warm-up", 1, 50)
    print("--> Img2Img Server is up and running!")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=False)
