import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    return model

# Example usage:
model = alexnet(pretrained=True)
print(model)
    
# Example usage of AlexNet

from torchvision.transforms import transforms
from PIL import Image

# 修改图像转换pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),  # 确保只使用前3个通道
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载图像
img = Image.open('image.png').convert('RGB')  # 确保转换为RGB

# 应用转换
img_tensor = transform(img).unsqueeze(0)  # 添加批次维度

# Create the model
model = alexnet(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Perform inference
with torch.no_grad():
    output = model(img_tensor)

# Get the predicted class
_, predicted_idx = torch.max(output, 1)

# If you have a list of class names, you can map the index to a class name
# class_names = [...] # List of 1000 ImageNet class names
# predicted_class = class_names[predicted_idx.item()]

print(f"Predicted class index: {predicted_idx.item()}")
# print(f"Predicted class: {predicted_class}")

# If you want to get the top-5 predictions
top5_prob, top5_idx = torch.topk(output, 5)
print("Top 5 predictions:")
for i in range(5):
    print(f"  {i+1}: Index {top5_idx[0][i].item()}, Probability: {top5_prob[0][i].item():.4f}")

