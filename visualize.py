import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ================== Global style ==================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14


# ================== Model definition ==================
class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.fc(x.view(batch_size, -1))
        return x


# ================== Transform ==================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


# ================== Load test data ==================
def load_test_data(test_root):
    images, labels = [], []

    for name in os.listdir(os.path.join(test_root, "NORMAL")):
        images.append(os.path.join(test_root, "NORMAL", name))
        labels.append(1)

    for name in os.listdir(os.path.join(test_root, "PNEUMONIA")):
        images.append(os.path.join(test_root, "PNEUMONIA", name))
        labels.append(0)

    return images, np.array(labels)


# ================== Prediction ==================
def predict_all(model, image_paths, device):
    probs = []
    model.eval()

    with torch.no_grad():
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = transform(img).unsqueeze(0).to(device)

            logit = model(img).view(-1)
            prob = torch.sigmoid(logit).item()
            probs.append(prob)

    return np.array(probs)


# ================== Confusion Matrix ==================
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    barbie_pink = LinearSegmentedColormap.from_list(
    "barbie_pink",
    ["#fde0ef", "#f783ac", "#e64980", "#c2255c"]
)

    norm = Normalize(vmin=cm.min(), vmax=cm.max())

    plt.imshow(cm, cmap=barbie_pink, norm=norm)
    plt.colorbar()


    classes = ["PNEUMONIA", "NORMAL"]
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center",
                     fontsize=1, fontweight="bold")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ================== ROC & AUC ==================
def plot_roc_curve(y_true, probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#e75480", linewidth=2,
             label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return roc_auc


# ================== Prediction Examples ==================
def visualize_examples(image_paths, y_true, probs, save_path, num=6):
    idxs = np.random.choice(len(image_paths), num, replace=False)

    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(idxs):
        img = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
        pred = 1 if probs[idx] >= 0.5 else 0

        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        plt.title(
            f"GT: {'NORMAL' if y_true[idx] == 1 else 'PNEUMONIA'}\n"
            f"Pred: {'NORMAL' if pred == 1 else 'PNEUMONIA'}\n"
            f"Prob: {probs[idx]:.2f}",
            fontsize=13
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ================== Main ==================
if __name__ == "__main__":
    device = torch.device("cpu")

    model = torch.load("model.pt", map_location=device)

    test_root = "chest_xray/test"
    image_paths, y_true = load_test_data(test_root)

    probs = predict_all(model, image_paths, device)
    y_pred = (probs >= 0.5).astype(int)

    os.makedirs("results2", exist_ok=True)

    plot_confusion_matrix(y_true, y_pred, "results2/confusion_matrix.png")
    auc_value = plot_roc_curve(y_true, probs, "results2/roc_curve.png")
    visualize_examples(image_paths, y_true, probs, "results2/prediction_examples.png")

    print(f"Visualization completed. AUC = {auc_value:.3f}")
