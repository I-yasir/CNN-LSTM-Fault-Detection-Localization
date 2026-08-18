import os
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models import FaultCNNLSTM
from src.datasets import UniversalFaultDataset
from src.utils import ensure_dir

def main():
    print("--- Step 3: Training Model ---")
    with open("configuration/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = np.load(config['paths']['splits'])
    
    train_dataset = UniversalFaultDataset(config['paths']['raw_data'], config['paths']['norm_params'], splits['train'])
    val_dataset = UniversalFaultDataset(config['paths']['raw_data'], config['paths']['norm_params'], splits['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=2)

    model = FaultCNNLSTM(in_channels=config['system']['input_channels'], num_classes=config['system']['num_classes']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_loc = nn.MSELoss(reduction='none') 
    
    best_val_loss = float('inf')
    epochs = config['training']['epochs']
    
    # History tracker for Loss Curve
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x, y_cls, y_loc in train_loader:
            x, y_cls, y_loc = x.to(device), y_cls.to(device), y_loc.to(device)
            optimizer.zero_grad()
            pred_cls, pred_loc = model(x)
            
            loss_cls = criterion_cls(pred_cls, y_cls)
            mask = (y_cls > 0).float()
            valid_faults = mask.sum()
            
            loss_loc = (criterion_loc(pred_loc, y_loc) * mask).sum() / valid_faults if valid_faults > 0 else torch.tensor(0.0).to(device)
            loss = loss_cls + (config['training']['loc_loss_weight'] * loss_loc)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_cls, y_loc in val_loader:
                x, y_cls, y_loc = x.to(device), y_cls.to(device), y_loc.to(device)
                pred_cls, pred_loc = model(x)
                loss_cls = criterion_cls(pred_cls, y_cls)
                mask = (y_cls > 0).float()
                valid_faults = mask.sum()
                loss_loc = (criterion_loc(pred_loc, y_loc) * mask).sum() / valid_faults if valid_faults > 0 else 0
                val_loss += (loss_cls + config['training']['loc_loss_weight'] * loss_loc).item()

        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f} - Val: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config['paths']['model_save'])
            
    print(f"Model saved to {config['paths']['model_save']}.\n")
    
    # Save training history
    ensure_dir(config['paths']['reports_dir'])
    with open(os.path.join(config['paths']['reports_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    main()