from .train_model import TrainModel
from .test_model import TestModel

def create_model(opt):
    model = None
    if opt.model == 'test':
        assert (opt.dataset_mode == 'test')
        model = TestModel(opt)
    else:
        model = TrainModel(opt)
    print("model [%s] was created" % (model.name()))
    return model