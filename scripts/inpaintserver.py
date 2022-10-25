from flask import Flask, request, jsonify, json
from flask_cors import CORS, cross_origin

# Initialize Redis
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Flask App
app = Flask(__name__)
CORS(app)

# Flask Routes
@app.route("/inpaint", methods=["POST"])
@cross_origin()
def inpaint_api():
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

@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)

# Flask Context
with app.app_context():
    print("--> Stable Diffusion Inpainting Server is up and running!")

# Main
if __name__ == "__main__":
    app.run()
