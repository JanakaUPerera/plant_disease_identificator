import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app_server import app

client = TestClient(app)

def jpeg(w, h, seed=0):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()

info = client.get("/api/info").json()
print("info: crops", list(info["crops"]), "| tomato classes", len(info["crops"]["tomato"]), "| model", info["model"])

cases = [
    ("normal tomato photo",       "tomato", ("leaf.jpg", jpeg(400, 300), "image/jpeg")),
    ("normal corn photo",         "corn",   ("leaf.jpg", jpeg(400, 300), "image/jpeg")),
    ("low resolution (150 px)",   "apple",  ("small.jpg", jpeg(180, 150), "image/jpeg")),
    ("unknown crop",              "banana", ("leaf.jpg", jpeg(400, 300), "image/jpeg")),
    ("not an image (text file)",  "tomato", ("notes.txt", b"hello, this is not a picture", "text/plain")),
    ("empty file",                "tomato", ("empty.jpg", b"", "image/jpeg")),
    ("too small (60 px)",         "tomato", ("tiny.jpg", jpeg(80, 60), "image/jpeg")),
    ("too large (11 MB)",         "tomato", ("big.jpg", b"\xff" * (11 * 1024 * 1024), "image/jpeg")),
]
for name, crop, f in cases:
    r = client.post("/api/predict", data={"crop": crop}, files={"file": f})
    body = r.json()
    if r.status_code == 200:
        top = body["results"][0]
        print(f"{name:26s} -> 200 | top: {top['name']} {top['probability']:.2f} ({top['type']}) | "
              f"{len(body['results'])} classes, sum {sum(x['probability'] for x in body['results']):.3f} | "
              f"warnings: {len(body['warnings'])} | {body['latency_ms']} ms")
        for w in body["warnings"]:
            print(f"{'':29s}! {w[:95]}")
    else:
        print(f"{name:26s} -> {r.status_code} | {body['detail']}")