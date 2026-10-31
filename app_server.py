"""
Web application backend (Question 4): the trained ANN answers each request at request time.

Start it from the plant_audit folder:
    uvicorn app_server:app --port 8000
then open http://127.0.0.1:8000 in a browser.

Endpoints
  GET  /api/info      crops, their classes, model details and limitations (for the page to show)
  POST /api/predict   form fields: crop (text) and file (an image); returns the crop's classes ranked by
                      calibrated probability, the recommended action, and any warnings
  POST /api/explain   form fields: crop, file and optionally class_id; returns a Grad-CAM heatmap of the
                      regions that raised that class's score (default: the most likely class)
"""
import base64
import io
import json
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from gradcam import GradCAM, overlay
from plantdata import eval_transform, load_classes
from plantmodel import CropMaskedResNet

ROOT = Path(__file__).parent
RUN_ID = "R15"                      # the deployed model, fixed before the final test
MAX_BYTES = 10 * 1024 * 1024        # largest upload accepted (10 MB)
MIN_SIDE = 100                      # same rule as data cleaning (R1): smaller images are rejected
LOW_RES_SIDE = 224                  # below the model's input size: accepted, with a warning
LOW_CONFIDENCE = 0.50               # calibrated top probability below this -> "uncertain" warning
LOW_MARGIN = 0.20                   # ... or the top answer less than 20 points ahead of the second

# what each class means for the officer's decision (Question 1)
CLASS_INFO = {
    "apple_scab":                ("Apple scab", "fungal"),
    "apple_cedar_rust":          ("Cedar apple rust", "fungal"),
    "apple_healthy":             ("Healthy apple leaf", "healthy"),
    "pepper_bacterial_spot":     ("Bell pepper bacterial spot", "bacterial"),
    "pepper_healthy":            ("Healthy bell pepper leaf", "healthy"),
    "corn_gray_leaf_spot":       ("Corn grey leaf spot", "fungal"),
    "corn_northern_leaf_blight": ("Corn northern leaf blight", "fungal"),
    "corn_common_rust":          ("Corn common rust", "fungal"),
    "grape_black_rot":           ("Grape black rot", "fungal"),
    "grape_healthy":             ("Healthy grape leaf", "healthy"),
    "potato_early_blight":       ("Potato early blight", "fungal"),
    "potato_late_blight":        ("Potato late blight", "oomycete"),
    "tomato_bacterial_spot":     ("Tomato bacterial spot", "bacterial"),
    "tomato_early_blight":       ("Tomato early blight", "fungal"),
    "tomato_late_blight":        ("Tomato late blight", "oomycete"),
    "tomato_leaf_mold":          ("Tomato leaf mould", "fungal"),
    "tomato_septoria_leaf_spot": ("Tomato septoria leaf spot", "fungal"),
    "tomato_mosaic_virus":       ("Tomato mosaic virus", "viral"),
    "tomato_yellow_leaf_curl":   ("Tomato yellow leaf curl virus", "viral"),
    "tomato_healthy":            ("Healthy tomato leaf", "healthy"),
}
ACTIONS = {
    "fungal":    "Treatment: consider a fungicide programme, following local extension guidelines.",
    "oomycete":  "Treatment, urgently: late blight spreads within days; apply a protective fungicide and "
                 "check neighbouring plants.",
    "bacterial": "Treatment: remove affected leaves; consider a copper-based bactericide, following local guidelines.",
    "viral":     "No cure: remove infected plants and control the insect vectors that spread the virus.",
    "healthy":   "No action needed.",
}
# Measured on the final test (results/final_test/, R15 = this model), so the officer sees real numbers.
# Only the words shown to the officer changed after the test; the model and its thresholds did not.
LIMITATIONS = [
    "Trained only on laboratory photos (PlantVillage). On unseen field photos (PlantWild test set) it named the "
    "right class about half the time (disease macro-F1 0.44); on unseen lab photos, 97% of the time.",
    "It over-calls disease on field leaves: about half of the healthy leaves in the field test got a disease answer "
    "(only about 1 in 100 diseased leaves was called healthy). Check for real symptoms before treating.",
    "Late-blight answers were right only about 1 time in 3 on field test photos (it caught 83% of real late "
    "blight). Confirm the signs before an urgent spray.",
    "Virus photos (tomato mosaic, yellow leaf curl) were answered as a fungal disease more than half the time; "
    "fungicide does not help against a virus.",
    "For photos of a single leaf. Fruit, stems, whole fields and other crops are outside what it was built for.",
    "Corn and potato have no 'healthy' class: for these crops the model always names a disease.",
    "Probabilities are calibrated (temperature scaling on validation photos); on the field test photos the "
    "average confidence matched the accuracy closely. They remain estimates: a 60% answer is wrong about 4 times in 10.",
    "A decision-support tool: confirm the diagnosis before treating, especially when the top answer is below 50%.",
]
CROP_NAMES = {"apple": "Apple", "bell_pepper": "Bell pepper", "corn": "Corn (maize)", "grape": "Grape",
              "potato": "Potato", "tomato": "Tomato"}

# ---------------------------------------------------------------- load the model once, at start-up
CLASS_IDS, CROPS, CROP_MASK = load_classes(ROOT / "splits_v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_ck = torch.load(ROOT / "checkpoints" / f"{RUN_ID}.pt", map_location="cpu", weights_only=False)
MODEL = CropMaskedResNet(_ck["config"]["depth"], CROP_MASK, pretrained=False)
MODEL.load_state_dict(_ck["state_dict"])
MODEL.to(DEVICE).eval()
TEMPERATURE = json.loads((ROOT / "checkpoints" / f"{RUN_ID}_calibration.json").read_text())["temperature"]
OOD = json.loads((ROOT / "checkpoints" / f"{RUN_ID}_ood.json").read_text())   # score name + threshold (val only)
TRANSFORM = eval_transform()
EXPLAINER = GradCAM(MODEL)          # Question 5: heatmaps from the last convolutional stage (layer4)
# warm-up: the first pass on a GPU is slow (kernels are loaded and tuned), so do it before the first officer
_dummy = torch.zeros(1, 3, 224, 224, device=DEVICE)
with torch.no_grad():
    MODEL(_dummy, torch.tensor([0], device=DEVICE))
EXPLAINER(_dummy, torch.tensor([0], device=DEVICE))
LIMITATIONS.append(
    f"The 'unfamiliar photo' check catches only about {1 - OOD['ood_missed']:.0%} of photos of other crops "
    f"(validation). No warning does NOT mean the photo is within what the model knows.")


def familiarity(logits, probs):
    """The out-of-distribution score chosen on validation data (higher = more familiar)."""
    return {"msp": float(probs.max()), "max_logit": float(logits.max()),
            "energy": float(torch.logsumexp(logits, dim=0))}[OOD["score"]]

app = FastAPI(title="Leaf disease assistant")


def crop_classes(crop):
    return [CLASS_IDS[j] for j in CROP_MASK[CROPS.index(crop)].nonzero().flatten().tolist()]


@app.get("/api/info")
def info():
    return {
        "crops": {crop: [{"id": c, "name": CLASS_INFO[c][0]} for c in crop_classes(crop)] for crop in CROPS},
        "crop_names": {crop: CROP_NAMES.get(crop, crop) for crop in CROPS},
        "model": {"run_id": RUN_ID, "architecture": f"ResNet-{_ck['config']['depth']} (fine-tuned)",
                  "temperature": round(TEMPERATURE, 3), "device": DEVICE,
                  "ood_score": OOD["score"], "ood_auroc": round(OOD["auroc"], 3)},
        "limitations": LIMITATIONS,
    }


async def checked_photo(crop, file):
    """The uploaded photo as an RGB image, after the same checks for every endpoint."""
    if crop not in CROPS:
        raise HTTPException(400, f"Unknown crop '{crop}'. Choose one of: {', '.join(CROPS)}.")
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file is larger than 10 MB. Please upload a smaller photo.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            image = ImageOps.exif_transpose(im).convert("RGB")
    except Image.DecompressionBombError:
        raise HTTPException(413, "The image has too many pixels to process safely.")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The file is not a readable image (JPEG, PNG or similar).")
    if min(image.size) < MIN_SIDE:
        raise HTTPException(422, f"The image is too small ({image.size[0]}x{image.size[1]} pixels). "
                                 f"Please upload a photo at least {MIN_SIDE} pixels on each side.")
    return image


@app.post("/api/predict")
async def predict(crop: str = Form(...), file: UploadFile = File(...)):
    start = time.perf_counter()
    image = await checked_photo(crop, file)
    x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    crop_idx = torch.tensor([CROPS.index(crop)], device=DEVICE)
    with torch.no_grad():
        logits = MODEL(x, crop_idx).float().cpu()[0]
    probs = (logits / TEMPERATURE).softmax(dim=0)

    ranked = sorted(crop_classes(crop), key=lambda c: -float(probs[CLASS_IDS.index(c)]))
    results = [{"id": c, "name": CLASS_INFO[c][0], "type": CLASS_INFO[c][1],
                "probability": round(float(probs[CLASS_IDS.index(c)]), 4),
                "action": ACTIONS[CLASS_INFO[c][1]]} for c in ranked]

    score = familiarity(logits, probs)
    unfamiliar = score < OOD["threshold"]
    warnings = []
    if unfamiliar:
        warnings.append("Unfamiliar photo: it looks unlike the photos the model learned from (perhaps another "
                        "crop, a fruit or not a leaf). The suggestions below are unreliable.")
    top, second = results[0]["probability"], results[1]["probability"]
    if top < LOW_CONFIDENCE:
        warnings.append(f"Uncertain: the most likely answer has only {top:.0%}. "
                        "Treat it as a suggestion and confirm before acting.")
    elif top - second < LOW_MARGIN:
        warnings.append(f"Uncertain: the top two answers are close ({top:.0%} and {second:.0%}). "
                        "Treat it as a suggestion and confirm before acting.")
    if results[0]["type"] == "oomycete":
        warnings.append("Late blight is urgent but often over-called: on field test photos only about 1 in 3 "
                        "late-blight answers was right. Before spraying, look for its signs: dark, water-soaked "
                        "patches, often with a pale edge, and white growth on the underside in damp weather.")
    if not any(c.endswith("_healthy") for c in ranked):
        warnings.append(f"The model cannot answer 'healthy' for {crop}: it always names one of the "
                        f"{len(ranked)} diseases, even for a healthy leaf.")
    if min(image.size) < LOW_RES_SIDE:
        warnings.append(f"Low resolution ({image.size[0]}x{image.size[1]}): small lesions may not be visible "
                        "to the model.")
    return {"crop": crop, "results": results, "warnings": warnings,
            "familiarity": {"score": OOD["score"], "value": round(score, 3), "threshold": round(OOD["threshold"], 3),
                            "unfamiliar": unfamiliar},
            "image_size": list(image.size), "model": RUN_ID,
            "latency_ms": round(1000 * (time.perf_counter() - start), 1)}


def data_url(picture):
    """A PIL image as text the browser can show directly: <img src="data:image/jpeg;base64,...">."""
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@app.post("/api/explain")
async def explain(crop: str = Form(...), file: UploadFile = File(...), class_id: str | None = Form(None)):
    start = time.perf_counter()
    image = await checked_photo(crop, file)
    if class_id is not None and class_id not in crop_classes(crop):
        raise HTTPException(400, f"'{class_id}' is not one of the classes for {crop}.")
    x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    crop_idx = torch.tensor([CROPS.index(crop)], device=DEVICE)
    cam, k, logits = EXPLAINER(x, crop_idx, class_idx=None if class_id is None else CLASS_IDS.index(class_id))
    probs = (logits.float() / TEMPERATURE).softmax(dim=0)
    view, picture = overlay(image, cam, size=448)
    return {"class_id": CLASS_IDS[k], "name": CLASS_INFO[CLASS_IDS[k]][0],
            "probability": round(float(probs[k]), 4),
            "heatmap": data_url(picture), "view": data_url(view),
            "method": "Grad-CAM, last convolutional stage (layer4)", "model": RUN_ID,
            "latency_ms": round(1000 * (time.perf_counter() - start), 1)}


# the web page (frontend) is served from the "static" folder at the site root
if (ROOT / "static").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")