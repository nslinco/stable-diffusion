import time
import base64
from io import BytesIO
from flask import json

from slavehelper import encodeImgs, postResponse
from inpaintmodel import InpaintModel as SDModel

# Initialize Redis
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Initialize Model
sd_model = None

# Perform Action
def doInpaint(job, model):
    # Parse request
    jobId = str(job['_id'])
    jobData = job['data']['inputs']

    image = jobData["image"].split(',')[1]
    mask = jobData["mask"].split(',')[1]
    num_steps = jobData["num_steps"]

    # Generate Images
    generated_img = model.generate_image(BytesIO(base64.b64decode(image)), BytesIO(base64.b64decode(mask)), num_steps)

    # Encode Images
    returned_generated_images = encodeImgs([generated_img])
    
    # Return Images
    return postResponse(jobId, {
        "generatedImgs": returned_generated_images,
        "generatedImgsFormat": "jpeg"
    })

# Main
def main():
    #Initialize Model
    print("--> Starting Stable Diffusion Inpaint Slave. This might take up to two minutes.")
    sd_model = SDModel()

    # Run Warm-Up Tests
    # sd_model.generate_images("warm-up", 1, 50)
    print("--> Stable Diffusion Inpaint Slave is up and running!")

    # Initialize Redis jobs
    jobs = json.dumps({ 'jobs': [] })
    r.set('jobs', jobs)

    # Enter Work Loop
    while(True):
        curJobs = json.loads(r.get('jobs'))['jobs']
        if(len(curJobs) > 0):
            curJob = curJobs[0]
            doneJob = doInpaint(curJob, sd_model)
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
