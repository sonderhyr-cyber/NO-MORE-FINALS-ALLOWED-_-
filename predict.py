import torch
import cv2
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# ===== 必须：和训练时一模一样的模型 =====
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


# ===== 图像预处理（和训练一致）=====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


def predict(image_path, model, device):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img).view(-1)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).int().item()

    return pred, prob.item()


# ===== 主程序 =====
if __name__ == "__main__":
    device = torch.device("cpu")  # Mac 先用 CPU

    model = torch.load("model.pt", map_location=device)
    model.eval()

    image_path = "test.jpeg"  # 换成你的 X 光 jpeg
    pred, prob = predict(image_path, model, device)

    label_map = {1: "NORMAL", 0: "PNEUMONIA"}

    print("预测结果:", label_map[pred])
    print("置信度:", f"{prob:.4f}")

