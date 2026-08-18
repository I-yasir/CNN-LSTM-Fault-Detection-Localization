import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_loss_curve(history_file, filepath):
    if not os.path.exists(history_file): 
        return
    with open(history_file, 'r') as f:
        history = json.load(f)
        
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Training Loss', lw=2)
    plt.plot(history['val_loss'], label='Validation Loss', lw=2)
    plt.title('Model Convergence (Loss Curve)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_probs, num_classes, filepath, class_names=None):
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    plt.figure(figsize=(10, 8))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        label = f'Class {i}' if class_names is None else class_names[i]
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) per Class')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, filepath, normalize=False):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Fault Classification Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_localization_scatter(y_true, y_pred, filepath, line_length):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.4, color='crimson', edgecolors='none')
    plt.plot([0, line_length], [0, line_length], color='black', linestyle='--', lw=2)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel('True Fault Location (% Line Length)')
    plt.ylabel('Predicted Fault Location (% Line Length)')
    plt.title('Localization Precision')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_sensitivity(df, x_col, acc_col, f1_col, filepath, title):
    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[acc_col] * 100, marker='o', linewidth=2, label='Accuracy')
    plt.plot(df[x_col], df[f1_col] * 100, marker='s', linestyle='--', linewidth=2, label='Macro F1')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel('Performance (%)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()