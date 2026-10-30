"""
Zero-shot plant disease recognition with pretrained vision-language models (Question 6).

A CLIP-style model has an image encoder and a text encoder that map into the same space. Without any
training on our data, a photo is classified by comparing its embedding with the embeddings of one text
prompt per class ("a photo of a tomato leaf with late blight.") and choosing the closest. As in the ANN,
the officer's crop restricts the answer to that crop's classes.
"""
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# words for each crop: (common name, scientific name)
CROP_TEXT = {
    "apple": ("apple", "Malus domestica"),
    "bell_pepper": ("bell pepper", "Capsicum annuum"),
    "corn": ("corn", "Zea mays"),
    "grape": ("grape", "Vitis vinifera"),
    "potato": ("potato", "Solanum tuberosum"),
    "tomato": ("tomato", "Solanum lycopersicum"),
}
# words for each disease (None = healthy)
DISEASE_TEXT = {
    "apple_scab": "apple scab", "apple_cedar_rust": "cedar apple rust", "apple_healthy": None,
    "pepper_bacterial_spot": "bacterial spot", "pepper_healthy": None,
    "corn_gray_leaf_spot": "gray leaf spot", "corn_northern_leaf_blight": "northern leaf blight",
    "corn_common_rust": "common rust",
    "grape_black_rot": "black rot", "grape_healthy": None,
    "potato_early_blight": "early blight", "potato_late_blight": "late blight",
    "tomato_bacterial_spot": "bacterial spot", "tomato_early_blight": "early blight",
    "tomato_late_blight": "late blight", "tomato_leaf_mold": "leaf mold",
    "tomato_septoria_leaf_spot": "septoria leaf spot", "tomato_mosaic_virus": "mosaic virus",
    "tomato_yellow_leaf_curl": "yellow leaf curl virus", "tomato_healthy": None,
}
# prompt templates: (for a disease, for a healthy leaf); {crop}, {sci} and {disease} are filled in
TEMPLATES = {
    "label":      ("{crop} {disease}", "healthy {crop}"),
    "photo":      ("a photo of {crop} {disease}.", "a photo of a healthy {crop} leaf."),
    "leaf":       ("a photo of a {crop} leaf with {disease}.", "a photo of a healthy {crop} leaf."),
    "symptoms":   ("a close-up photo of a {crop} leaf showing symptoms of {disease}, a plant disease.",
                   "a close-up photo of a healthy green {crop} leaf with no disease."),
    "scientific": ("a photo of a leaf of {crop} ({sci}) with {disease}.",
                   "a photo of a healthy leaf of {crop} ({sci})."),
}
TEMPLATE_SETS = {name: [name] for name in TEMPLATES}
TEMPLATE_SETS["ensemble"] = ["photo", "leaf", "symptoms", "scientific"]   # average of four prompts per class


def prompt(class_id, crop, template):
    """The text for one class with one template."""
    disease_fmt, healthy_fmt = TEMPLATES[template]
    common, sci = CROP_TEXT[crop]
    disease = DISEASE_TEXT[class_id]
    return (healthy_fmt if disease is None else disease_fmt).format(crop=common, sci=sci, disease=disease)


@torch.no_grad()
def text_features(clip, class_ids, class_crops, template_set, device="cuda"):
    """One unit-length embedding per class (classes x embedding size). With several templates the
    embeddings are averaged and normalised again (prompt ensembling)."""
    model, tokenizer = clip["model"], clip["tokenizer"]
    per_template = []
    for template in TEMPLATE_SETS[template_set]:
        texts = [prompt(c, crop, template) for c, crop in zip(class_ids, class_crops)]
        emb = model.encode_text(tokenizer(texts).to(device)).float()
        per_template.append(emb / emb.norm(dim=-1, keepdim=True))
    mean = torch.stack(per_template).mean(dim=0)
    return mean / mean.norm(dim=-1, keepdim=True)


class PhotoDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = list(paths), preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.preprocess(im.convert("RGB"))


@torch.no_grad()
def image_features(clip, paths, device="cuda", batch_size=64, num_workers=0):
    """One unit-length embedding per photo (photos x embedding size), using the model's own preprocessing."""
    loader = DataLoader(PhotoDataset(paths, clip["preprocess"]), batch_size=batch_size, num_workers=num_workers)
    out = []
    for images in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            emb = clip["model"].encode_image(images.to(device))
        emb = emb.float()
        out.append((emb / emb.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


def zero_shot_logits(img_feats, txt_feats, crop_idx, crop_mask, logit_scale):
    """Similarity of every photo to every class, scaled as in training; other crops' classes excluded."""
    logits = logit_scale * img_feats @ txt_feats.T
    return logits.masked_fill(~crop_mask[crop_idx], float("-inf"))