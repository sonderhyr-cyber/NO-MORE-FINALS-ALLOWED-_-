import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 预处理数据
def _preprocess(output, label):
    output = output.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    output = np.where(output >= 0.5, 1, 0)
    return output, label

def calculate_accuracy(output, label):
    output, label = _preprocess(output, label)
    return accuracy_score(label, output)

def calculate_precision(output, label):
    output, label = _preprocess(output, label)
    return precision_score(label, output, zero_division=0)

def calculate_recall(output, label):
    output, label = _preprocess(output, label)
    return recall_score(label, output, zero_division=0)

def calculate_f1(output, label):
    output, label = _preprocess(output, label)
    return f1_score(label, output, zero_division=0)