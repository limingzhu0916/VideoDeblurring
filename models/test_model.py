from base_model import BaseModel
import networks
import torch
from torch.autograd import Variable
from loss import init_loss

class TestModel(BaseModel):
    def name(self):
        return 'TrainModel'

    def __init__(self, opt):
        BaseModel.__init__(self, opt)