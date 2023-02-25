import comet_ml

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torchmetrics import JaccardIndex
from torchvision import datasets
import argparse

args = {
    'exp_name': 'starter',
    'data_root': '../../../vcg_natural/cityscape',
    'model': 'tu-xception65',
    'batch_size': 4, # For DP, this is the total batch size; for ddp this is for each GPU core
    'parallel_mode': 'dp',
    'crop_size': 768, # Crop size for training the model
    'num_classes': 20, # 19 effective classes, index=19 is ignored
    'n_iters': 60000,
    'lr': 0.007, # This is scaled by number of GPUs
    'n_cpus': 16,
    'epochs': 20
}

'''parser = argparse.ArgumentParser(description='Placeholder')

parser.add_argument("--exp_name", type=str, default='starter')
parser.add_argument("--dara_root", type=str, default='.')
parser.add_argument("--lr", type=float, default=1.2e-4)
parser.add_argument("--model", type=str, default='tu-xception65')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--parallel_mode", type=str, default='ddp')
parser.add_argument("--crop_size", type=int, default=768)
parser.add_argument("--num_classes", type=int, default=20)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--n_cpus", type=int, default=16)

args = vars(parser.parse_args())'''

args['lr'] *= torch.cuda.device_count()
# For data-parallel (jupyter notebook), args['batch_size'] denotes actual batch size split across different 
#  devices. For ddp (script mode), args['batch_size'] is the batch size for a single GPU.
if args['parallel_mode'] == 'dp':
    args['batch_size'] *= torch.cuda.device_count()

if not os.path.exists('logs'):
    os.mkdir('logs')
    
# Change comet api credentials for different users

comet_logger = pl.loggers.CometLogger(
    api_key=os.environ.get("COMET_API_KEY"),
    workspace=os.environ.get("COMET_WORKSPACE"),  # Optional
    save_dir="logs",  # Optional
    project_name="active_learning",  # Optional
    experiment_name=args['exp_name']  # Optional
)

comet_logger.log_hyperparams(args)
    
args['log_dir'] = os.path.join('tkaminsky', 'logs', args['exp_name'])
model = smp.DeepLabV3Plus(encoder_name=args['model'], encoder_weights='imagenet', \
                          in_channels=3, classes=args['num_classes'], activation=None,\
                          decoder_atrous_rates=(6, 12, 18))

print(args)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
target_transform = transforms.Compose([
    transforms.ToTensor(),
    # Multiply by 255 to offset the effects of `ToTensor`
    transforms.Lambda(lambda x : (x * 255).to(torch.long))])

class CityScapesDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, random_crop_size=None, augment=False):
        self.dataset = dataset 
        # for mask postprocessing
        valid_classes = list(filter(lambda x : x.ignore_in_eval == False, self.dataset.classes))
        self.class_names = [x.name for x in valid_classes] + ['void']
        self.id_map = {old_id : new_id for (new_id, old_id) in enumerate([x.id for x in valid_classes])}
        self.crop_size = random_crop_size 
        self.augment = augment
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        for cur_id in label.unique():
            cur_id = cur_id.item()
            if cur_id not in self.id_map.keys():
                label[label == cur_id] = 250 
            else:
                label[label == cur_id] = self.id_map[cur_id] 
        label[label == 250] = 19
        label = label.squeeze(0)
        
        # Random cropping
        if self.crop_size is not None:
            H, W = img.shape[1:]
            h, w = np.random.randint(0, H - self.crop_size), np.random.randint(0, W - self.crop_size)
            img, label = map(lambda x : x[..., h : h + self.crop_size, w : w + self.crop_size], [img, label])
        # Random horizontal flip
        if self.augment and np.random.rand() < .5:
            img, label = map(transforms.functional.hflip, [img, label])
        return {'inputs': img, 'label': label}

trainset = CityScapesDatasetWrapper(datasets.Cityscapes(root=args['data_root'],\
                               split='train', mode='fine', target_type='semantic',\
                               transform=img_transform, target_transform=target_transform, transforms=None),\
                                random_crop_size=args['crop_size'], augment=True)
valset = CityScapesDatasetWrapper(datasets.Cityscapes(root=args['data_root'],\
                               split='val', mode='fine', target_type='semantic', \
                               transform=img_transform, target_transform=target_transform, transforms=None))
testset = CityScapesDatasetWrapper(datasets.Cityscapes(root=args['data_root'],\
                               split='test', mode='fine', target_type='semantic', \
                               transform=img_transform, target_transform=target_transform, transforms=None))

epoch_steps = len(trainset) // args['batch_size']
if args['parallel_mode'] == 'ddp':
    epoch_steps = len(trainset) // (args['batch_size'] * torch.cuda.device_count())
print(len(trainset), len(valset), len(testset), epoch_steps)


from torchmetrics.functional import jaccard_index

def jaccard_fn(preds, labels):
    labels = labels.to(preds.get_device())
    metric_value = jaccard_index(preds, labels, num_classes=20, average='macro', ignore_index=19)
    return metric_value 

class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.trainset, self.valset, self.testset = trainset, valset, testset
        self.model = model
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=19)
        self.metric_fn = jaccard_fn # JaccardIndex(num_classes=20, ignore_index=19)
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_nb):
        preds = self.forward(batch['inputs'])
        loss_val = self.criterion(preds, batch['label'])
        metric_val = self.metric_fn(preds, batch['label'].to(preds.get_device()))
        self.log('loss', loss_val, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log('metric', metric_val, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss' : loss_val, 'metric': metric_val}
    
    def validation_step(self, batch, batch_nb) :
        preds = self.forward(batch['inputs'])
        loss_val = self.criterion(preds, batch['label'])
        metric_val = self.metric_fn(preds, batch['label'].to(preds.get_device()))
        self.log('val_loss', loss_val, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log('val_metric', metric_val, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        return {'val_loss' : loss_val, 'val_metric': metric_val}
    
    def configure_optimizers(self):
        op = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=.9, weight_decay=0.00004)
        scheduler = torch.optim.lr_scheduler.LambdaLR(op, lambda step : (1 - step / args['n_iters']) ** .9)
        return {"optimizer": op, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.args['batch_size'], pin_memory=True,\
                                           shuffle=True, drop_last=True, num_workers=args['n_cpus'])
    
    def val_dataloader(self):
        # Smaller batch size for validation because the validation size is 1024 x 2048
        return torch.utils.data.DataLoader(self.valset, batch_size=self.args['batch_size'] // 2, drop_last=True,
                                           num_workers=args['n_cpus'], pin_memory=True)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, batch_size=self.args['batch_size'], \
                                           num_workers=args['n_cpus'])


model = SegmentationModel()
earlystopping_callback = pl.callbacks.EarlyStopping(monitor='val_metric', mode='max', patience=8)
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args['log_dir'], filename='model.ckpt', \
                                                   monitor='val_metric')
lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=False)

trainer = pl.Trainer(accelerator='gpu', strategy=args['parallel_mode'], precision=16,
                     default_root_dir=args['log_dir'], benchmark=False,\
                     max_epochs=args['epochs'], log_every_n_steps=1,\
                     logger=comet_logger, callbacks=[earlystopping_callback, lr_logger, checkpoint_callback])
trainer.fit(model)

