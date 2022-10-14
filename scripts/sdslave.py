import time
import base64
from io import BytesIO
import requests
from flask import jsonify

from sdmodel import SDModel

# Initialize Redis
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Initialize Model
sd_model = None

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

def main();
    while(true):
        curJobs = r.get('jobs')
        if(len(curJobs) > 0):
            curJob = curJobs[0]
            doneJob = doSD(curJob)
            if(!doneJob):
                print('Job Error:', curJob['_id'])
            else
                print('Job Complete:', curJob['_id'])
                jobs = r.get('jobs')
                jobs.remove(curJob)
                r.set('jobs', jobs)
        time.sleep(1)
