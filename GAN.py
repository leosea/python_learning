import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
batch_size = 64
lr = 0.0002
epochs = 50
latent_dim = 100

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# 初始化模型
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 损失函数和优化器
adversarial_loss = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# 训练循环
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # 准备真实图像和标签
        real_imgs = imgs.to(device)
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        # 训练判别器
        d_optimizer.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), real_labels)
        
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器
        g_optimizer.zero_grad()
        g_loss = adversarial_loss(discriminator(fake_imgs), real_labels)
        g_loss.backward()
        g_optimizer.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # 每个epoch结束后保存生成的图像
    if epoch % 10 == 0:
        z = torch.randn(16, latent_dim).to(device)
        generated_imgs = generator(z).detach().cpu()
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(generated_imgs[i*4+j].squeeze(), cmap='gray')
                axs[i, j].axis('off')
        plt.savefig(f'generated_imgs_epoch_{epoch}.png')
        plt.close()

print("训练完成!")
