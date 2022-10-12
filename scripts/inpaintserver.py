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
print("--> Starting Stable Diffusion Inpainting Server. This might take up to two minutes.")

from inpaintmodel import InpaintModel
inpaint_model = None

curJobs = {
    'inpaint': None
}

parser = argparse.ArgumentParser(description = "A Stable Diffusion app to turn your textual prompts into visionary delights")
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

def doInpaint():
    # Parse request
    job = curJobs['inpaint']
    jobId = job['_id']
    jobData = job['data']
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

@app.route("/inpaint", methods=["POST"])
@cross_origin()
def inpaint_api():
    # Return if busy
    if (curJobs['inpaint']): return jsonify(curJobs['inpaint']['_id'], 'waiting')

    # Parse request
    job = request.get_json(force=True)

    # Run Job
    curJobs['inpaint'] = job
    Thread(target = doInpaint()).start()
    
    # Report Job has started
    return jsonify(job['_id'], 'working')


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
    # Initialize Models
    inpaint_model = InpaintModel()

    # Run Warm-Up Tests
    # t1 = inpaint_model.generate_image(BytesIO(base64.b64decode(tImg)), BytesIO(base64.b64decode(tMask)), 50)
    print("--> Stable Diffusion Inpainting Server is up and running!")


if __name__ == "__main__":
    app.run()
