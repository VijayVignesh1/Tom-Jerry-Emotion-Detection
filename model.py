from imports import *
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 5)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        #print("out",out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        # print("out",out.shape)
        return out

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=6):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        
        alexnet = torchvision.models.alexnet(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(alexnet.children())[:-1]
        self.alexnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # model=torch.nn.Sequential(*module,,adaptive_pool, , 
        #                   torch.nn.Linear(9216,4096), torch.nn.ReLU(), torch.nn.Dropout(0.5), )
        self.max=torch.nn.MaxPool2d(3,2)
        self.dropout=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(9216,4096)
        self.fc2=torch.nn.Linear(4096,1024)
        self.fc3=torch.nn.Linear(1024,128)
        self.fc4=torch.nn.Linear(128,5)

        
    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.alexnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out=self.max(out)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        batch_size=out.size(0)
        #print(out.shape)
        out=out.reshape(batch_size,-1,9216)
        out=self.dropout(out)
        # out = out.view(batch_size, -1, 9216)
        out = self.fc1(out)
        out=self.dropout(out)
        out=self.fc2(out)
        out=self.dropout(out)
        out=self.fc3(out)
        out=self.dropout(out)
        out=self.fc4(out)
        #print(out.shape)
        return out.reshape(batch_size,5)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet=torchvision.models.resnet50(pretrained=True)
        self.fc=torch.nn.Linear(1000,5)
    def forward(self,image):
        out=self.resnet(image)  
        out=self.fc(out)
        return out

class Ensemble(nn.Module):
    """
    Encoder.
    """

    def __init__(self):
        super(Ensemble, self).__init__()
        self.resnet1=torchvision.models.resnet18(pretrained=True)
        self.resnet2=torchvision.models.resnet34(pretrained=True)
        self.resnet3=torchvision.models.resnet50(pretrained=True)
        self.resnet4=torchvision.models.resnet101(pretrained=True)
        self.resnet5=torchvision.models.resnet152(pretrained=True)
        self.fc=torch.nn.Linear(1000,5)
        # self.dropout=torch.nn.Dropout(0.5)
    def forward(self,image):
        #out1=self.resnet1(image)
        #out2=self.resnet2(image)
        out3=self.resnet3(image)
        out4=self.resnet4(image)
        #out5=self.resnet5(image)
        # out=torch.cat([out1,out2,out3],dim=1)
        out=torch.mean(torch.stack([out3,out4],dim=0).float(),dim=0)
        # print(out.shape)
        out=self.fc(out)
        # out=self.dropout(out)
        return out