import time
import base64
from io import BytesIO
import requests
from flask import jsonify, json

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
    x = requests.post('https://feedback-backend.herokuapp.com/create/callback', json=myobj)
    return (x)

def doSD(job, model):
    # Parse request
    print('doSD job: ', job)
    jobId = str(job['_id'])
    jobData = job['data']
    text_prompt = jobData["prompt"]
    num_images = jobData["num_images"]
    num_steps = jobData["num_steps"]

    # Generate Images
    generated_imgs = model.generate_images(text_prompt, num_images, num_steps)

    # Encode Images
    returned_generated_images = encodeImgs(generated_imgs)
    
    # Return Images
    return postResponse(jobId, {
        "generatedImgs": returned_generated_images,
        "generatedImgsFormat": "jpeg"
    })

def main():
    #Initialize Model
    print("--> Starting Stable Diffusion Slave. This might take up to two minutes.")
    sd_model = SDModel()

    # Run Warm-Up Tests
    sd_model.generate_images("warm-up", 1, 50)
    print("--> Stable Diffusion Slave is up and running!")

    # Initialize Redis jobs
    jobs = json.dumps({ 'jobs': [] })
    r.set('jobs', jobs)

    # Enter Work Loop
    while(True):
        curJobs = json.loads(r.get('jobs'))['jobs']
        if(len(curJobs) > 0):
            curJob = curJobs[0]
            doneJob = doSD(curJob, sd_model)
            if(not doneJob):
                print('Job Error:', curJob['_id'])
            else:
                print('Job Complete:', curJob['_id'])
                jobs = json.loads(r.get('jobs'))['jobs']
                jobs.remove(curJob)
                jobs = json.dumps({ 'jobs': jobs})
                r.set('jobs', jobs)
        time.sleep(1)

if __name__ == "__main__":
    main()
