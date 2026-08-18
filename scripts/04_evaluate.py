import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from src.models import FaultCNNLSTM
from src.datasets import UniversalFaultDataset
from src.utils import (ensure_dir, plot_confusion_matrix, plot_localization_scatter, 
                       plot_sensitivity, plot_loss_curve, plot_roc_curve)

def evaluate_subset(loader, model, device, feature_name=None):
    all_y_cls, all_pred_cls, all_pred_probs = [], [], []
    all_y_loc, all_pred_loc = [], []
    all_features = []
    
    with torch.no_grad():
        for batch in loader:
            x, y_cls, y_loc = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits, pred_loc = model(x)
            
            probs = torch.softmax(logits, dim=1)
            preds_c = torch.argmax(probs, dim=1)
            
            all_y_cls.extend(y_cls.cpu().numpy())
            all_pred_cls.extend(preds_c.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())
            all_y_loc.extend(y_loc.cpu().numpy())
            all_pred_loc.extend(pred_loc.cpu().numpy())
            
            if feature_name and len(batch) > 3:
                all_features.extend(batch[3].numpy())

    return (np.array(all_y_cls), np.array(all_pred_cls), np.array(all_pred_probs), 
            np.array(all_y_loc), np.array(all_pred_loc), np.array(all_features))

def main():
    print("--- Step 4: Full Pipeline Evaluation ---")
    with open("configuration/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = config['system']['num_classes']
    
    model = FaultCNNLSTM(in_channels=config['system']['input_channels'], num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(config['paths']['model_save'], map_location=device, weights_only=True))
    model.eval()

    reports_dir = config['paths']['reports_dir']
    ensure_dir(reports_dir)
    
    # Optional: Map classes to real names if you want nice labels in ROC/CM
    class_names = ['No Fault', 'AG', 'BG', 'CG', 'AB', 'BC', 'CA', 'ABG', 'BCG', 'CAG', 'ABCG']
    # Fallback if class names don't match num_classes
    if len(class_names) != num_classes:
        class_names = [f'Class {i}' for i in range(num_classes)]

    # Plot Loss Curve from training history
    history_file = os.path.join(reports_dir, 'training_history.json')
    plot_loss_curve(history_file, os.path.join(reports_dir, 'loss_curve.png'))

    # 1. Standard Test Evaluation
    print("Evaluating standard test split...")
    splits = np.load(config['paths']['splits'])
    test_ds = UniversalFaultDataset(config['paths']['raw_data'], config['paths']['norm_params'], splits['test'])
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False)
    
    y_true, y_pred, y_probs, loc_true, loc_pred, _ = evaluate_subset(test_loader, model, device)
    
    # A. Generate Visualizations
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(reports_dir, 'confusion_matrix.png'), normalize=True)
    plot_roc_curve(y_true, y_probs, num_classes, os.path.join(reports_dir, 'roc_curve.png'), class_names)
    
    mask = y_true > 0
    if mask.any():
        plot_localization_scatter(loc_true[mask], loc_pred[mask], os.path.join(reports_dir, 'localization_scatter.png'), config['system']['line_length_km'])
        mae = mean_absolute_error(loc_true[mask], loc_pred[mask])
        rmse = np.sqrt(mean_squared_error(loc_true[mask], loc_pred[mask]))
        r2 = r2_score(loc_true[mask], loc_pred[mask])
    else:
        mae, rmse, r2 = 0.0, 0.0, 0.0

    # B. Generate Tabular Reports (CSV & TXT)
    # 1. Detailed Classification Report (.csv)
    cls_report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(cls_report_dict).transpose().to_csv(os.path.join(reports_dir, 'classification_report.csv'))
    
    # 2. Comprehensive Thesis Summary (.txt)
    with open(os.path.join(reports_dir, 'thesis_metrics_summary.txt'), 'w') as f:
        f.write("========================================================\n")
        f.write("      MULTI-TASK FAULT MODEL EVALUATION REPORT          \n")
        f.write("========================================================\n\n")
        
        f.write("--- 1. Classification Performance ---\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4) + "\n\n")
        
        f.write("--- 2. Localization Performance (Faults Only) ---\n")
        f.write(f"Mean Absolute Error (MAE):       {mae:.4f} km\n")
        f.write(f"Root Mean Squared Error (RMSE):  {rmse:.4f} km\n")
        f.write(f"Coefficient of Determination (R2): {r2:.4f}\n")
        
        if config['system']['line_length_km'] > 0:
            error_percent = (mae / config['system']['line_length_km']) * 100
            f.write(f"Average Relative Error (%):      {error_percent:.4f} %\n")

    # 2. SNR Evaluation (Optional robustness check)
    if os.path.exists(config['paths']['snr_data']):
        print("Evaluating SNR robustness...")
        snr_ds = UniversalFaultDataset(config['paths']['snr_data'], config['paths']['norm_params'], extra_feature='SNR')
        snr_loader = DataLoader(snr_ds, batch_size=config['training']['batch_size'], shuffle=False)
        y_true_s, y_pred_s, _, _, _, features = evaluate_subset(snr_loader, model, device, 'SNR')
        
        df = pd.DataFrame({'True_Class': y_true_s, 'Pred_Class': y_pred_s, 'SNR': features})
        snr_metrics = [{'SNR': snr, 'Accuracy': accuracy_score(sub['True_Class'], sub['Pred_Class']), 
                        'F1': f1_score(sub['True_Class'], sub['Pred_Class'], average='macro')}
                       for snr, sub in df.groupby('SNR')]
        pd.DataFrame(snr_metrics).to_csv(os.path.join(reports_dir, 'snr_metrics.csv'), index=False)
        plot_sensitivity(pd.DataFrame(snr_metrics), 'SNR', 'Accuracy', 'F1', os.path.join(reports_dir, 'snr_sensitivity.png'), 'SNR Robustness')

    # 3. Resistance Evaluation (Optional robustness check)
    if os.path.exists(config['paths']['resistance_data']):
        print("Evaluating Resistance robustness...")
        res_ds = UniversalFaultDataset(config['paths']['resistance_data'], config['paths']['norm_params'], extra_feature='fault_resistance')
        res_loader = DataLoader(res_ds, batch_size=config['training']['batch_size'], shuffle=False)
        y_true_r, y_pred_r, _, _, _, features = evaluate_subset(res_loader, model, device, 'fault_resistance')
        
        df = pd.DataFrame({'True_Class': y_true_r, 'Pred_Class': y_pred_r, 'Res': features})
        res_metrics = [{'Resistance': res, 'Accuracy': accuracy_score(sub['True_Class'], sub['Pred_Class']), 
                        'F1': f1_score(sub['True_Class'], sub['Pred_Class'], average='macro')}
                       for res, sub in df.groupby('Res')]
        pd.DataFrame(res_metrics).to_csv(os.path.join(reports_dir, 'resistance_metrics.csv'), index=False)
        plot_sensitivity(pd.DataFrame(res_metrics), 'Resistance', 'Accuracy', 'F1', os.path.join(reports_dir, 'res_sensitivity.png'), 'Resistance Robustness')

    print(f"Evaluation complete. Check the '{reports_dir}' directory for all plots and tabular reports.")

if __name__ == "__main__":
    main()