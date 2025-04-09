import torch
import numpy as np
from tqdm import tqdm
from model import CNNModel
import torch.nn.functional as F
import pickle
import os

def train_model(opt, train_dl, valid_dl):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: opt.lamda ** e)
    criterion = torch.nn.CrossEntropyLoss()

    dict_result = {'train_loss': [], 'valid_loss': [], 'stop_point': 0}
    best_loss = float('inf')
    early_stop_count = 0
    best_model = None

    pbar = tqdm(range(opt.epochs), desc="Training")

    for epoch in pbar:
        model.train()
        train_loss = []
        for x, y in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        scheduler.step()
        
        pbar.set_postfix({'train_loss': np.mean(train_loss)})

        # Validation
        model.eval()
        val_loss = []
        with torch.no_grad():
            for x, y in valid_dl:
                val_loss.append(criterion(model(x), y).item())

        train_avg = np.mean(train_loss)
        valid_avg = np.mean(val_loss)
        dict_result['train_loss'].append(train_avg)
        dict_result['valid_loss'].append(valid_avg)

        if valid_avg < best_loss:
            best_loss = valid_avg
            early_stop_count = 0
            dict_result['stop_point'] = epoch + 1
            best_model = model
            
        else:
            early_stop_count += 1
            if early_stop_count > opt.early_stop:
                break

    return best_model, dict_result