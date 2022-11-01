import base64
from io import BytesIO
import requests
from flask import json

# Convert image arrays into base64-encoded image strings
def encodeImgs(generated_imgs):
    returned_generated_images = []    
    for idx, img in enumerate(generated_imgs):
        buffered = BytesIO()
        img.save(buffered, format='jpeg')
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_str = f'data:image/jpeg;base64,{img_str}'
        returned_generated_images.append(img_str)
    return (returned_generated_images)

# Call backend to report finished job
def getRequest():
    x = requests.get('https://feedback-backend.herokuapp.com/create/ready', headers = {})
    return (x)

# Call backend to report finished job
def postResponse(jobId, response):
    myobj = {}
    myobj[jobId] = response
    x = requests.post('https://feedback-backend.herokuapp.com/create/callback', headers = {}, json = myobj)
    return (x)

# Call backend to report finished job
def postUpdate(jobId, updateObj):
    myobj = {}
    myobj[jobId] = updateObj
    x = requests.post('https://feedback-backend.herokuapp.com/create/update', headers = {}, json = myobj)
    return (x)