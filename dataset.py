import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).unsqueeze(1).float()
        self.y = torch.argmax(torch.tensor(y), 1).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(device), self.y[idx].to(device)

def get_dataloaders(X, Y, opt):
    x_train, x_valid, y_train, y_valid = train_test_split(
        X, Y, train_size=opt.train_size, stratify=Y, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_valid, y_valid, train_size=0.5, stratify=y_valid, random_state=42)

    train_dl = DataLoader(MyDataset(x_train, y_train), batch_size=opt.batch_size, shuffle=True)
    valid_dl = DataLoader(MyDataset(x_valid, y_valid), batch_size=opt.batch_size, shuffle=False)
    test_dl = DataLoader(MyDataset(x_test, y_test), batch_size=opt.batch_size, shuffle=False)
    
    return train_dl, valid_dl, test_dl
