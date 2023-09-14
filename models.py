import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor
import torch.nn as nn
from torchvision.models.resnet import resnet18


class FasterRCNN():
    def __init__(self, weights, device):
        self.device = device
        self.model = self.get_model(weights)

        self.transform = Compose([
            ToPILImage(),
            Resize((320, 320)),
            ToTensor()
        ])

    def get_model(self, weights):
        model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(weights))
        model.eval()
        model.to(self.device)
        return model

    def preprocessing(self, image):
        return self.transform(image)

    def predict(self, image_tensor):
        with torch.no_grad():
            prediction = self.model([image_tensor])
        return prediction[0]


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=4)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))


class Resnet():
    def __init__(self, weights, device):
        self.device = device
        self.model = self.get_model(weights)
        self.classes = ["niezahartowany", "brak_wielowypustu", "brak_kanalka", "srodek_nieobrobiony"]

        self.transform = Compose([
            ToPILImage(),
            Resize((128, 128)),
            ToTensor(),
        ])

    def get_model(self, weights):
        model = Resnet18()
        model.load_state_dict(torch.load(weights))
        model.eval()
        model.to(self.device)
        return model

    def preprocessing(self, image):
        return self.transform(image)

    def predict(self, image_tensor, threshold):
        with torch.no_grad():
            prediction = self.model(image_tensor)[0]

        result_tensor = (prediction > threshold).float()
        output = [self.classes[i] for i in range(len(self.classes)) if result_tensor[i] > threshold]
        return output
