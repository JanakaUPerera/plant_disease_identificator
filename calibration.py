"""
Temperature scaling (Question 4): makes the displayed probabilities more honest.

A fine-tuned network is usually overconfident: when it says "90%", it is right less than 90% of the
time. Temperature scaling divides every logit by one number T (> 1 softens, < 1 sharpens) before the
softmax. T is fitted on val_wild only; it never changes which class is ranked first, only how sure
the model claims to be.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_logits(model, loader, device="cuda"):
    """Raw (crop-masked) logits and true classes for every image in a loader."""
    model.eval()
    logits, labels = [], []
    for images, crop_b, y in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            out = model(images.to(device, non_blocking=True), crop_b.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels):
    """The T that minimises cross-entropy (negative log-likelihood) of softmax(logits / T)."""
    log_t = torch.zeros(1, requires_grad=True)          # optimise log T so T stays positive
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimiser.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_t.detach().exp())


def expected_calibration_error(probs, labels, n_bins=15):
    """Average gap between confidence and accuracy, over confidence bins weighted by their size."""
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.float().mean() * (conf[in_bin].mean() - correct[in_bin].mean()).abs()
    return float(ece)


def reliability(probs, labels, n_bins=10):
    """Per confidence bin: mean confidence, accuracy and number of images (for a reliability diagram)."""
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            rows.append((float(conf[in_bin].mean()), float(correct[in_bin].mean()), int(in_bin.sum())))
    return rows