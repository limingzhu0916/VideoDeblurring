import torch.utils.data

def CreateDataset(opt):
    dataset = None
    validation = None
    if opt.dataset_mode == 'train':
        from data.train_data import TrainDataset
        dataset = TrainDataset(opt)
        if opt.use_validation:
            from data.test_data import TestDataset
            validation = TestDataset()
            validation.initialize(opt)
    elif opt.dataset_mode == 'test':
        from data.test_data import TestDataset
        dataset = TestDataset()
        dataset.initialize(opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    return dataset, validation

class BaseDataLoader():
    def __init__(self):
        pass
    def initialize(self, opt):
        self.opt = opt
        pass
    def load_data():
        return None


class CustomDatasetDataLoader(BaseDataLoader):
    """Wrapper class of Dataset class that performs multi-threaded data loading"""
    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, opt):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        super(CustomDatasetDataLoader, self).initialize(opt)
        print("Opt.nThreads = ", opt.nThreads)
        self.dataset_mode = opt.dataset_mode
        self.use_validation = opt.use_validation
        self.dataset, self.validation = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads)
        )
        if self.dataset_mode == 'train' and opt.use_validation:
            self.validation = torch.utils.data.DataLoader(
                self.validation,
                batch_size=opt.batchSize,
                shuffle=opt.serial_batches,
                num_workers=1
            )


    def load_data(self):
        if self.dataset_mode == 'train' and self.use_validation:
            return self.dataloader, self.validation
        else:
            return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader(opt)
    print(data_loader.name())
    return data_loader
