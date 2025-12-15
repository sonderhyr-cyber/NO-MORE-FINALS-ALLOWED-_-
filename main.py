import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import torchvision.models as models
from torchvision import transforms
import logging
import cv2
import matplotlib.pyplot as plt
from function import *

logging.basicConfig(level=logging.INFO)

# 模型主体
class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉fc层
        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Dropout(0.4),
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.fc(x.view(batch_size, -1))
        return x

# 数据集
class Xray_Dataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = torch.tensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = cv2.imread(self.features[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self.transform:
            return self.transform(img), self.labels[idx]
        else:
            return img, self.labels[idx]
    
    def get_data_size(self):
        return len(self.features)

# 分类器
class Classifier():
    def __init__(self, batch_size, data_root, transform, lr, epoch, check_point_num, freez_num):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'使用设备:{self.device} 开始训练')

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0], device=self.device))
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.transform = transform
        self.check_point_num = check_point_num
        self.freez_num = freez_num

        logging.info('开始创建模型')
        self.model = Resnet18().to(self.device)

        self.optimizer = None
        self.scheduler = None
        self.init_optimizer()

        self.freeze_backbone()  # 初始冻结

        logging.info('开始加载数据集')
        self.train_dataloader, self.test_dataloader = self.create_dataloader(data_root)
    
    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )

    def create_dataloader(self, data_root):
        train_path = os.path.join(data_root, 'train')
        test_path = os.path.join(data_root, 'test')

        train_data, train_label = self.get_image_labels(train_path)
        test_data, test_label = self.get_image_labels(test_path)

        train_dataset = Xray_Dataset(train_data, train_label, transform=self.transform)
        test_dataset = Xray_Dataset(test_data, test_label, transform=self.transform)

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        logging.info(f'数据集创建完毕,训练集大小:{train_dataset.get_data_size()} 测试集大小:{test_dataset.get_data_size()}')
        
        return train_dataloader, test_dataloader
    
    def get_image_labels(self, path):
        images = []
        labels = []
        # 加载正常图像
        for root, dirs, files in os.walk(os.path.join(path, 'NORMAL')):
            for name in files:
                images.append(os.path.join(root, name))
                labels.append(1)
        
        # 加载异常图像
        for root, dirs, files in os.walk(os.path.join(path, 'PNEUMONIA')):
            for name in files:
                images.append(os.path.join(root, name))
                labels.append(0)

        return images, labels
    
    def freeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def train(self):

        self.Train_Loss_Log = []
        self.Test_Loss_Log = []

        self.Train_Acc_Log = []
        self.Test_Acc_Log = []

        self.Train_Precision_Log = []
        self.Test_Precision_Log = []

        self.Train_Recall_Log = []
        self.Test_Recall_Log = []

        self.Train_F1_Log = []
        self.Test_F1_Log = []

        for epoch in range(1, self.epoch + 1):
            # 解冻判断
            if epoch == self.freez_num + 1:
                self.unfreeze_backbone()
                logging.info(f"Epoch {epoch}: 解冻backbone，开始微调")

                new_lr = self.lr * 0.1  # 解冻后学习率降为原来的1/10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                # 设置调度器
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epoch - self.freez_num,  # 解冻后剩余的epoch数
                    eta_min=new_lr * 0.01  # 最小学习率
                )

            total_loss = 0.0
            total_acc = 0.0
            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            
            for index, (image, label) in enumerate(self.train_dataloader):
                image, label = image.to(self.device), label.to(self.device).to(torch.float)
                
                self.optimizer.zero_grad()
                output = self.model(image).view(-1).to(torch.float)

                loss = self.criterion(output, label).to(self.device)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() 

                # 计算评价指标
                acc = calculate_accuracy(output, label)
                precision = calculate_precision(output, label)
                recall = calculate_recall(output, label)
                f1 = calculate_f1(output, label)

                total_acc += acc
                total_precision += precision
                total_recall += recall
                total_f1 += f1

                print(f'\repoch:{epoch} [{index}/{len(self.train_dataloader)}] Loss:{loss.item():.4f} Accuracy: {acc:.4f} Precision:{precision:.4f} Recall:{recall:.4f} F1:{f1:.4f}', end='')
            
            eval_loss, eval_acc, eval_precision, eval_recall, eval_f1 = self.eval()

            total_loss /= len(self.train_dataloader)
            total_acc /= len(self.train_dataloader)
            total_precision /= len(self.train_dataloader)
            total_recall /= len(self.train_dataloader)
            total_f1 /= len(self.train_dataloader)

            print(f'\repoch:{epoch} [{len(self.train_dataloader)}/{len(self.train_dataloader)}] Loss:{total_loss:.4f} Accuracy: {total_acc:.4f} Precision:{total_precision:.4f} Recall:{total_recall:.4f} F1:{total_f1:.4f}', 
                  f'\n\t\teval_Loss:{eval_loss:.4f} eval_Accuracy:{eval_acc:.4f} eval_Precision:{eval_precision:.4f} eval_Recall:{eval_recall:.4f} eval_F1:{eval_f1:.4f}')
            
            self.Train_Loss_Log.append(total_loss)
            self.Test_Loss_Log.append(eval_loss)

            self.Train_Acc_Log.append(total_acc)
            self.Test_Acc_Log.append(eval_acc)

            self.Train_Precision_Log.append(total_precision)
            self.Test_Precision_Log.append(eval_precision)

            self.Train_Recall_Log.append(total_recall)
            self.Test_Recall_Log.append(eval_recall)

            self.Train_F1_Log.append(total_f1)
            self.Test_F1_Log.append(eval_f1)
            if epoch % self.check_point_num == 0:
                self.check_point(epoch)
            

    def eval(self):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for index, (image, label) in enumerate(self.test_dataloader): 
            image, label = image.to(self.device), label.to(self.device)
            
            output = self.model(image).view(-1)
            loss = self.criterion(output.to(torch.float), label.to(torch.float)).to(self.device)
            total_loss += loss.item()

            acc = calculate_accuracy(output, label)
            precision = calculate_precision(output, label)
            recall = calculate_recall(output, label)
            f1 = calculate_f1(output, label)

            total_acc += acc
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        total_loss /= len(self.test_dataloader)
        total_acc /= len(self.test_dataloader)
        total_precision /= len(self.test_dataloader)
        total_recall /= len(self.test_dataloader)
        total_f1 /= len(self.test_dataloader)

        return total_loss, total_acc, total_precision, total_recall, total_f1
    
    def draw_line_chart_loss(self, save_path=None):

        epochs = list(range(1, len(self.Train_Loss_Log) + 1))

        plt.figure(figsize=(8, 6))
        plt.title('LOSS', fontsize=14)
        plt.xlabel('EPOCH', fontsize=12)
        plt.ylabel('LOSS', fontsize=12)
        plt.plot(epochs, self.Train_Loss_Log, label='train', marker='o', linewidth=2)
        plt.plot(epochs, self.Test_Loss_Log, label='eval', marker='s', linewidth=2)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        if save_path:
            plt.savefig(save_path)
            logging.info(f'图像已保存为{save_path}')
        else:
            plt.show()
        
        plt.close()
        
    def draw_line_chart(self, save_path=None):

        epochs = list(range(1, len(self.Train_Acc_Log) + 1))

        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 准确率
        axes[0, 0].plot(epochs, self.Train_Acc_Log, 'b-', label='Train Acc', linewidth=2)
        axes[0, 0].plot(epochs, self.Test_Acc_Log, 'r-', label='Test Acc', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Training and Testing Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 精确率
        axes[0, 1].plot(epochs, self.Train_Precision_Log, 'b-', label='Train Precision', linewidth=2)
        axes[0, 1].plot(epochs, self.Test_Precision_Log, 'r-', label='Test Precision', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Training and Testing Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 召回率
        axes[1, 0].plot(epochs, self.Train_Recall_Log, 'b-', label='Train Recall', linewidth=2)
        axes[1, 0].plot(epochs, self.Test_Recall_Log, 'r-', label='Test Recall', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Training and Testing Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # F1分数
        axes[1, 1].plot(epochs, self.Train_F1_Log, 'b-', label='Train F1', linewidth=2)
        axes[1, 1].plot(epochs, self.Test_F1_Log, 'r-', label='Test F1', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('Training and Testing F1-Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f'图像已保存为{save_path}')
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self, path=None):
        if path:
            torch.save(self.model, path)
        else:
            torch.save(self.model, 'model.pt')
    
    def check_point(self, epoch):
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
        self.save_model(path=f'checkpoint/check_point_epoch_{epoch}.pt')
        self.draw_line_chart_loss(save_path='checkpoint/line_chart_loss.png')
        self.draw_line_chart(save_path='checkpoint/line_chart.png')
        logging.info(f'检查点epoch:{epoch}设置完成')


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-b','--batch_size', type=int, help='训练批次大小', default=64)
    parser.add_argument('-e','--epoch', type=int, help='训练轮次', default=20)
    parser.add_argument('-l','--lr', type=float, help='训练速率', default=5e-6)
    parser.add_argument('-d','--data_root', type=str, help='训练数据根目录', default='chest_xray')
    parser.add_argument('-c','--check_point_num', type=int, help='每n步设置一个检查点检查点', default=2)
    parser.add_argument('-f','--freez_num', type=int, help='冻结backbone层训练的epoch数', default=5)

    args = parser.parse_args()

    logging.info('程序启动，基本超参数:')
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"    {key}: {value}  (type: {type(value).__name__})")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        #数据增强，防过拟合
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    classifier = Classifier(batch_size=args.batch_size, data_root=args.data_root, transform=transform, lr=args.lr,
                            epoch=args.epoch, check_point_num=args.check_point_num, freez_num=args.freez_num)
    classifier.train()
    classifier.draw_line_chart_loss(save_path='line_chart_loss.png')
    classifier.draw_line_chart(save_path='line_chart.png')
    classifier.save_model()

if __name__ == '__main__':
    main()