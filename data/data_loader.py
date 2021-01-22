import torch.utils.data

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'train':
        from data.train_data import DeblurDatset
        dataset = DeblurDatset(opt)
    # elif opt.dataset_mode == 'test':
        # from data.single_dataset import SingleDataset
        # dataset = SingleDataset()
        # dataset.initialize(opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    return dataset


class CustomDatasetDataLoader():
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
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads)
        )

    def load_data(self):
        if self.dataset_mode == 'train':
            return self.dataloader
        else:
            return self.dataloader

    def __len__(self):
        return len(self.dataset)

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader(opt)
    print(data_loader.name())
    return data_loader
