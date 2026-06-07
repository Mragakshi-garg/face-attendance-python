import os
import sys
import base64
import logging

import numpy as np
import cv2

from flask import Flask, request, jsonify
from flask_cors import CORS

from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load InsightFace model — buffalo_s is lighter (~30MB) and fits Render free tier
# ---------------------------------------------------------------------------

# logger.info("Loading InsightFace model (buffalo_s)...")

# face_app = FaceAnalysis(
#     name="buffalo_s",
#     providers=["CPUExecutionProvider"]
# )

# face_app.prepare(ctx_id=-1, det_size=(320,320))

# logger.info("InsightFace loaded successfully.")
face_app = None

def get_face_app():
    global face_app

    if face_app is None:
        logger.info("Loading InsightFace model...")

        face_app = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"]
        )

        face_app.prepare(
            ctx_id=-1,
            det_size=(320, 320)
        )

        logger.info("InsightFace loaded successfully.")

    return face_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_base64_image(b64_string):
    """
    Accepts:
        /9j/4AAQSk...
    OR
        data:image/jpeg;base64,/9j/4AAQSk...
    """

    try:

        if not b64_string:
            return None

        # Remove browser prefix if present
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(b64_string)

        np_arr = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception as e:

        logger.exception(
            f"Base64 decode failed: {e}"
        )

        return None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "service": "insightface"
    })


# ---------------------------------------------------------------------------
# Detect faces
# ---------------------------------------------------------------------------

@app.route("/detect", methods=["POST"])
def detect():

    try:

        data = request.get_json(force=True)

        if not data or "image" not in data:

            return jsonify({
                "success": False,
                "error": "Missing image"
            }), 400

        image = decode_base64_image(
            data["image"]
        )

        if image is None:

            logger.error(
                "Failed to decode image received from Node"
            )

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400

        logger.info(
            f"Image received successfully. Shape={image.shape}"
        )

        detected_faces = get_face_app().get(image)

        faces = []

        for face in detected_faces:

            bbox = face.bbox.astype(int)

            left = int(bbox[0])
            top = int(bbox[1])
            right = int(bbox[2])
            bottom = int(bbox[3])

            faces.append({
                "location": [
                    top,
                    right,
                    bottom,
                    left
                ],
                "encoding": face.embedding.tolist()
            })

        logger.info(
            f"Detected {len(faces)} face(s)"
        )

        return jsonify({
            "success": True,
            "faces": faces
        })

    except Exception as e:

        logger.exception("Detect error")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ---------------------------------------------------------------------------
# Encode face
# ---------------------------------------------------------------------------

@app.route("/encode", methods=["POST"])
def encode():

    try:

        if "photo" not in request.files:

            return jsonify({
                "success": False,
                "error": "No photo uploaded"
            }), 400

        file = request.files["photo"]

        file_bytes = file.read()

        np_arr = np.frombuffer(
            file_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400

        detected_faces = get_face_app().get(image)

        if len(detected_faces) == 0:

            return jsonify({
                "success": False,
                "error": "No face detected"
            })

        encoding = detected_faces[0].embedding.tolist()

        return jsonify({
            "success": True,
            "encoding": encoding
        })

    except Exception as e:

        logger.exception("Encode error")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ---------------------------------------------------------------------------
# Compare embeddings
# ---------------------------------------------------------------------------

@app.route("/compare", methods=["POST"])
def compare():

    try:
        data = request.get_json(force=True)

        encoding = data.get("encoding")
        known_encodings = data.get("known_encodings")
        known_ids = data.get("known_ids")

        if (
            encoding is None or
            known_encodings is None or
            known_ids is None
        ):

            return jsonify({
                "success": False,
                "error": "Missing required fields"
            }), 400

        if len(known_encodings) == 0:

            return jsonify({
                "success": True,
                "match": False,
                "student_id": None,
                "confidence": 0.0
            })

        target_encoding = np.array(encoding)

        known_encoding_arrays = [
            np.array(e)
            for e in known_encodings
        ]

        # Calculate cosine similarity for each known encoding
        similarities = []

        for known in known_encoding_arrays:
            norm_known = np.linalg.norm(known)
            norm_target = np.linalg.norm(target_encoding)

            if norm_known == 0 or norm_target == 0:
                similarities.append(0.0)
            else:
                cosine = float(
                    np.dot(known, target_encoding) / (norm_known * norm_target)
                )
                similarities.append(cosine)

        best_idx = int(np.argmax(similarities))
        best_similarity = similarities[best_idx]

        logger.info(
            f"Compare: best_similarity={best_similarity:.4f}, "
            f"best_idx={best_idx}"
        )

        # Threshold for cosine similarity match
        threshold = 0.4
        is_match = best_similarity > threshold
        confidence = round(best_similarity, 4)

        return jsonify({
            "success": True,
            "match": is_match,
            "student_id": (
                known_ids[best_idx]
                if is_match
                else None
            ),
            "confidence": confidence
        })

    except Exception as e:

        logger.exception(
            "Compare error"
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5001
        )
    )

    logger.info(
        f"Starting InsightFace service on port {port}"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )