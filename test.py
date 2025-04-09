import torch
import numpy as np

def evaluate(test_dl, model):
    
    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for x, y in test_dl:
            y_pred = model(x)
            preds.append(torch.argmax(y_pred, 1).cpu())
            labels.append(y.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = (preds == labels).mean() * 100
    return acc
