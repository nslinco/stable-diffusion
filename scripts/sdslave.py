import time
from flask import json

from slavehelper import encodeImgs, postResponse
from sdmodel import SDModel

# Initialize Redis
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Initialize Model
sd_model = None

def doSD(job, model):
    # Parse request
    jobId = str(job['_id'])
    jobData = job['data']['inputs']
    
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
