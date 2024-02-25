import os
from typing import Optional, Any
from aqueduct import BaseTask, BaseTaskHandler, Flow, FlowStep
# from nudenet import NudeClassifier
from tempfile import NamedTemporaryFile


import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from torch.autograd import Variable


CLASSES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']


class NSFWTask(BaseTask):
    tmp_file_path: Optional[str] = None
    img_class: Any

    def __init__(self, file_content):
        super(NSFWTask, self).__init__()
        self.tmp_file_path = ''
        self.file_content = file_content
        self.img_class = {}


class CheckNSFWHandler(BaseTaskHandler):
    MODEL_PATH = './nsfw_mobilenet2.224x224.h5'

    def on_start(self):
        self._trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        self._model = models.resnet50()
        self._model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1),
        )
        self._model.load_state_dict(
            torch.load('ResNet50_nsfw_model.pth', map_location=torch.device('cpu')),
        )
        self._model.eval()

    def handle(self, *tasks: NSFWTask):
        for task in tasks:
            image = Image.open(task.tmp_file_path)
            print(image)
            image_tensor = self._trans(image).float()
            image_tensor = image_tensor.unsqueeze_(0)

            input = Variable(image_tensor)
            output = self._model(input)
            res = output.data.numpy()[0]
            res = np.exp(res) / np.sum(np.exp(res))
            # res = np.e ^ (res - max(res)) / sum(np.e ^ (res - max(res)))
            print(res)
            task.img_class = {key: float(value) for key, value in zip(CLASSES, res)}


class StoreTempFileHandler(BaseTaskHandler):

    def handle(self, *tasks: NSFWTask):
        for task in tasks:
            with NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                task.tmp_file_path = f.name
                print(task.tmp_file_path)
                f.write(task.file_content)
                task.file_content = None


class ClearTempFileHandler(BaseTaskHandler):

    def handle(self, *tasks: NSFWTask):
        for task in tasks:
            os.remove(task.tmp_file_path)
            task.tmp_file_path = None


def build_flow():
    return Flow(
        FlowStep(StoreTempFileHandler()),
        FlowStep(CheckNSFWHandler()),
        FlowStep(ClearTempFileHandler()),
    )
