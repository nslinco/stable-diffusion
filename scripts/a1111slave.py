import time
from flask import json

import requests
import io
import base64
from PIL import Image

from slavehelper import postResponse, getRequest, getInstancePublicDNS
from sdmodel import SDModel

import boto3
client = boto3.client('s3', region_name='us-west-1')

# Initialize Redis
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Initialize Model
# sd_model = None

def doSD(job, instanceDNS):
    try:
        # Generate Images
        r.set('status', 'working')

        # Parse job
        jobId = job["parentId"]
        optjobUID = job["jobUID"]
        animate = job["animate"]
        optddim_steps = job["ddim_steps"]
        optprompt = job["prompt"]
        optmodifier = job["modifier"]
        optseed = job["seed"]
        optddim_eta = job["eta"]
        optscale = job["scale"]

        prompt = optprompt + ', ' + optmodifier
        
        url = "http://127.0.0.1:7860"

        payload = {
            "prompt": prompt,
            "steps": optddim_steps,
            "cfg_scale": optscale,
            "seed": optseed,
            "eta": optddim_eta
        }

        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

        response = response.json()

        generated_imgs = []
        for i in response['images']:
            print(f"Generating image for bulk job: {jobId}-{optjobUID}")

            # Name image
            gifName = f"{jobId}-{optjobUID}.gif"
            imgName = f"{jobId}-{optjobUID}.jpeg"
            
            tic = time.time()

            uTic = 0
            uToc = 0
            uTime = 0
            
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

            # Upload to s3 bucket
            # img = Image.fromarray(image.astype(np.uint8)) # Faster to leave output_type as default?
            uTic = time.time()
            buffered = io.BytesIO()
            image.save(fp=buffered, format='jpeg')
            buffered.seek(0)
            client.upload_fileobj(
                buffered,
                'meadowrun-sd-69',
                'images/{}'.format(imgName),
                ExtraArgs={'ACL':'public-read'}
            )
            uToc = time.time()
            uTime = uToc-uTic

            toc = time.time()
            iTime = toc-tic

            retObj = {
                "jobId": jobId,
                "jobUID": optjobUID,
                "modelId": 'a1111',
                "result": imgName,
                "params": {
                    "prompt": optprompt,
                    "modifier": optmodifier,
                    "seed": optseed,
                    "eta": optddim_eta,
                    "scale": optscale,
                    "ddim_steps": optddim_steps
                },
                "time": iTime,
                "animation": gifName
            }
            generated_imgs.append(retObj)

        print(f"Generated {len(generated_imgs)} images for bulk job: {jobId}-{optjobUID} in {iTime-uTime}-{uTime}s")
    
        r.set('status', 'waiting')

        # Report Results
        newJob = postResponse(job["parentId"], generated_imgs, instanceDNS)
        return (newJob)
    except Exception as e:
        print(f'doSD Error: {e}')
        r.set('status', 'failed')

def main():
    # Get instance public DNS
    instanceDNS = getInstancePublicDNS()
    print("instanceDNS:", instanceDNS)
    # r.set('instanceDNS', instanceDNS)

    #Initialize Model
    print("--> Starting Stable Diffusion Slave")
    r.set('status', 'initializing')
    # sd_model = SDModel()

    # Run Warm-Up Tests
    print("--> Stable Diffusion Slave is up and running!")

    # Initialize Redis jobs
    jobs = json.dumps({ 'jobs': [] })
    r.set('jobs', jobs)
    r.set('status', 'waiting')

    # Enter Work Loop
    r.set('sleepCounter', '0')
    while(True):
        try:
            curJobs = json.loads(r.get('jobs'))['jobs']
            if(len(curJobs) > 0):
                curJob = curJobs[0]
                print(f'Starting quick job: {curJob["_id"]}')
                newJob = doSD(curJob, instanceDNS)

                print('Job Complete:', curJob['_id'])
                jobs = json.loads(r.get('jobs'))['jobs']
                jobs.remove(curJob)
                if(newJob):
                    jobs.append(newJob)
                r.set('jobs', json.dumps({ 'jobs': jobs}))
            else:
                status = r.get('status').decode("utf-8")
                print('status: ', status)
                if (status != 'sleeping'):
                    newJob = getRequest(instanceDNS)
                    print("newJob: ", newJob)
                    if (newJob):
                        jobs = json.loads(r.get('jobs'))['jobs']
                        jobs.append(newJob)
                        r.set('jobs', json.dumps({'jobs': jobs}))
                    else:
                        sleepCounter = int(r.get('sleepCounter').decode('utf-8'))
                        sleepCounter += 1
                        if (sleepCounter > 2):
                            r.set('sleepCounter', 0)
                            r.set('status', 'sleeping')
                        else:
                            r.set('sleepCounter', sleepCounter)
        except Exception as e:
            print(f'Error!: {e}')
        time.sleep(5)

if __name__ == "__main__":
    main()
