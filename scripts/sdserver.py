from flask import Flask, request, jsonify, json
from flask_cors import CORS, cross_origin
# import argparse
# from consts import DEFAULT_IMG_OUTPUT_DIR
# from utils import parse_arg_boolean, parse_arg_dalle_version
# from consts import ModelSize

# Initialize Redis
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Flask App
app = Flask(__name__)
# app.config.update(CELERY_CONFIG={
#     'broker_url': 'redis://localhost:6379',
# })
CORS(app)

# parser = argparse.ArgumentParser(description = "A Stable Diffusion app to turn your textual prompts into visionary delights")
# parser.add_argument("--port", type=int, default=8000, help = "backend port")
# parser.add_argument("--model_version", type = parse_arg_dalle_version, default = ModelSize.MINI, help = "Mini, Mega, or Mega_full")
# parser.add_argument("--save_to_disk", type = parse_arg_boolean, default = False, help = "Should save generated images to disk")
# parser.add_argument("--img_format", type = str.lower, default = "JPEG", help = "Generated images format", choices=['jpeg', 'png'])
# parser.add_argument("--output_dir", type = str, default = DEFAULT_IMG_OUTPUT_DIR, help = "Customer directory for generated images")
# args, unknown = parser.parse_known_args()

# Flask Routes
@app.route("/sd", methods=["POST"])
@cross_origin()
def sd_api():
    # Parse request
    job = request.get_json(force=True)

    # Save to Redis
    jobs = json.loads(r.get('jobs'))['jobs']
    jobs.append(job)
    r.set('jobs', json.dumps({'jobs': jobs}))
    
    # Report Job has started
    res = {}
    res[job['_id']] = 'working'
    return jsonify(
        message=f"Working on {job['_id']}",
        data=res,
        status=200
    )

@app.route("/status", methods=["GET"])
@cross_origin()
def sd_status():
    status = r.get('status').decode("utf-8")
    jobs = json.loads(r.get('jobs'))['jobs']
    return jsonify(
        message=f"Here are my current jobs",
        data={
            "status": status,
            "jobs": jobs
        },
        status=200
    )

@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)

# Flask Context
with app.app_context():
    print("--> Stable Diffusion Gunicorn Server is up and running!")

# Main
if __name__ == "__main__":
    app.run()
