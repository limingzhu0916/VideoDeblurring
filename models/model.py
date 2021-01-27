from .train_model import TrainModel

def create_model(opt):
    model = None
    if opt.model == 'test':
        assert (opt.dataset_mode == 'test')
        from .test_model import TestModel
        model = TestModel(opt)
    else:
        model = TrainModel(opt)
    print("model [%s] was created" % (model.name()))
    return model