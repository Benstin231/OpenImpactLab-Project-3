import json
import threading
import uuid
from flask import Flask, render_template, request, Response, jsonify
from src import orchestrator

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/research", methods=["POST"])
def start_research():
    data = request.get_json(silent=True)
    if not data or not data.get("prompt", "").strip():
        return jsonify({"error": "prompt is required"}), 400

    prompt = data["prompt"].strip()
    request_id = str(uuid.uuid4())
    q = orchestrator.create_queue(request_id)

    thread = threading.Thread(
        target=orchestrator.run,
        args=(prompt, request_id, q),
        daemon=True,
    )
    thread.start()

    return jsonify({"request_id": request_id})


@app.route("/stream/<request_id>")
def stream(request_id: str):
    q = orchestrator.get_queue(request_id)
    if q is None:
        return jsonify({"error": "request not found"}), 404

    def generate():
        try:
            while True:
                event = q.get(timeout=120)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("event_type") == "done":
                    break
        except Exception:
            yield f"data: {json.dumps({'event_type': 'error', 'data': {'message': 'Stream timeout'}, 'timestamp': ''})}\n\n"
        finally:
            orchestrator.remove_queue(request_id)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
