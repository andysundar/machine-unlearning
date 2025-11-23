
import torch, numpy as np
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def confidence_scores(model, loader, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()
    scores = []
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).max(dim=1).values
        scores.append(probs.cpu().numpy())
    return np.concatenate(scores, axis=0)

def mia_auc(in_scores, out_scores):
    y_true = np.concatenate([np.ones_like(in_scores), np.zeros_like(out_scores)])
    y_score = np.concatenate([in_scores, out_scores])
    return roc_auc_score(y_true, y_score)
