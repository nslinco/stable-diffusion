import base64
from io import BytesIO
import requests
from flask import jsonify
from celery import Celery

from sdmodel import SDModel

# Make Celery
app = Celery('tasks', broker='redis://localhost:6379')

# Initialize Model
print("--> Starting Stable Diffusion Server. This might take up to two minutes.")
sd_model = SDModel()

# Run Warm-Up Tests
sd_model.generate_images("warm-up", 1, 50)

def encodeImgs(generated_imgs):
    returned_generated_images = []    
    for idx, img in enumerate(generated_imgs):
        buffered = BytesIO()
        img.save(buffered, format='jpeg')
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_str = f'data:image/jpeg;base64,{img_str}'
        returned_generated_images.append(img_str)
    return (returned_generated_images)

def postResponse(jobId, response):
    myobj = {}
    myobj[jobId] = response
    x = requests.post('https://feedback-backend.herokuapp.com/create/callback', data=myobj)
    return (x)

@app.task()
def doSD(job):
    # Parse request
    print('doSD job: ', job)
    jobId = job['_id']
    jobData = job['data']
    text_prompt = jobData["prompt"]
    num_images = jobData["num_images"]
    num_steps = jobData["num_steps"]

    # Generate Images
    generated_imgs = sd_model.generate_images(text_prompt, num_images, num_steps)

    # Encode Images
    returned_generated_images = encodeImgs(generated_imgs)
    
    # Return Images
    return postResponse(jobId, jsonify({
        'generatedImgs': returned_generated_images,
        'generatedImgsFormat': 'jpeg'
    }))

# def make_celery(app):
#     celery = Celery(app.import_name)
#     celery.conf.update(app.config["CELERY_CONFIG"])

#     class ContextTask(celery.Task):
#         def __call__(self, *args, **kwargs):
#             with app.app_context():
#                 return self.run(*args, **kwargs)

#     celery.Task = ContextTask
#     return celery