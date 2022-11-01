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
    
    prompt = jobData["prompt"]
    n_samples = jobData["n_samples"]
    ddim_steps = jobData["ddim_steps"]
    plms = jobData["plms"]
    fixed_code = jobData["fixed_code"]
    ddim_eta = jobData["ddim_eta"]
    n_iter = jobData["n_iter"]
    H = jobData["H"]
    W = jobData["W"]
    C = jobData["C"]
    f = jobData["f"]
    scale = jobData["scale"]
    precision = jobData["precision"]
    seed = jobData["seed"]

    # Generate Images
    generated_imgs = model.generate_images(
        prompt,
        n_samples,
        ddim_steps,
        plms,
        fixed_code,
        ddim_eta,
        n_iter,
        H,
        W,
        C,
        f,
        scale,
        precision,
        seed
    )

    # Encode Images
    returned_generated_images = encodeImgs(generated_imgs)
    
    # Return Images
    return postResponse(jobId, {
        "generatedImgs": returned_generated_images,
        "generatedImgsFormat": "jpeg"
    })

def doSDBulk(job, model):
    # Parse request
    jobId = str(job['_id'])
    instanceIp = job['instanceIp']
    jobData = job['data']['inputs']

    subJobs = jobData["subJobs"]
    ddim_steps = jobData["ddim_steps"]
    animate = jobData["animate"]
    mask = jobData["mask"]

    # Generate Images
    generated_imgs = model.generate_images_bulk(
        instanceIp,
        jobId,
        subJobs,
        ddim_steps,
        animate,
        mask
    )
    
    # Return Images
    return postResponse(jobId, {
        "generatedImgs": generated_imgs,
        "generatedImgsFormat": "jpeg",
        "bulk": True
    })

def doSDQuick(job, model):
    try:
        # Generate Images
        r.set('status', 'working')
        generated_imgs = model.generate_images_quick(job)
        r.set('status', 'waiting')

        # Report Results
        newJob = postResponse(job["parentId"], generated_imgs)
        if (newJob):
            # Save Next Job
            jobs = json.loads(r.get('jobs'))['jobs']
            jobs.append(newJob)
            r.set('jobs', json.dumps({'jobs': jobs}))
    except Exception as e:
        print(f'doSDQuick Error: {e}')
        r.set('status', 'failed')

def main():
    #Initialize Model
    print("--> Starting Stable Diffusion Slave. This might take up to two minutes.")
    r.set('status', 'initializing')
    sd_model = SDModel()

    # Run Warm-Up Tests
    sd_model.generate_images("warm-up")
    print("--> Stable Diffusion Slave is up and running!")

    # Initialize Redis jobs
    jobs = json.dumps({ 'jobs': [] })
    r.set('jobs', jobs)
    r.set('status', 'waiting')

    # Enter Work Loop
    while(True):
        try:
            curJobs = json.loads(r.get('jobs'))['jobs']
            if(len(curJobs) > 0):
                curJob = curJobs[0]
                doneJob = doSDQuick(curJob, sd_model)

                # doneJob = None
                # if (curJob['data']['bulk']):
                #     print(f'Starting bulk job: {curJob["_id"]}')
                #     doneJob = doSDBulk(curJob, sd_model)
                # else:
                #     print(f'Starting job: {curJob["_id"]}')
                #     doneJob = doSD(curJob, sd_model)
                
                if(not doneJob):
                    print('Job Error:', curJob['_id'])
                else:
                    print('Job Complete:', curJob['_id'])
                    jobs = json.loads(r.get('jobs'))['jobs']
                    jobs.remove(curJob)
                    jobs = json.dumps({ 'jobs': jobs})
                    r.set('jobs', jobs)
        except Exception as e:
            print(f'Error!: {e}')
        time.sleep(1)

if __name__ == "__main__":
    main()
