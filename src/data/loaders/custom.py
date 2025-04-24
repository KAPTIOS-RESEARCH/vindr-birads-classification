from torch.utils.data import Subset, DataLoader
from src.data.sets.custom import CustomImageFolder

class CustomImageDataloader(object):
    def __init__(self, 
                 data_dir: str, 
                 batch_size: int = 4,
                 image_size: int = 224,
                 augment_type: str = "geometric",
                 num_workers: int = 4,
                 debug: bool = True):
        
        super(CustomImageDataloader, self).__init__()
        self.data_dir = data_dir
        self.debug = debug
        self.image_size = image_size
        self.augment_type = augment_type
        self.batch_size = batch_size
        self.num_workers = num_workers 

    def train(self):
        train_dataset = CustomImageFolder(
            root=self.data_dir,
            image_size=self.image_size,
            train=True,
            augment_type=self.augment_type
        )
        
        if self.debug:
            train_dataset = Subset(train_dataset, range(self.batch_size * 2))
        
        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                pin_memory=True)
        return dataloader

    def val(self):
        val_dataset = CustomImageFolder(
            root=self.data_dir,
            image_size=self.image_size,
            train=False,
            augment_type=self.augment_type
        )
        
        if self.debug:
            val_dataset = Subset(val_dataset, range(self.batch_size * 2))
            
        dataloader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                pin_memory=True)
        return dataloader

    def test(self):
        return self.val()