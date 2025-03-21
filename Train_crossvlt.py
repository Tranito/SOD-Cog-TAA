from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from src.model import accident
from src.improved_model import improved_accident


from src.dataset import DADA
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import numpy as np
from src.bert import opt
import gc
from Test import validation, write_validation_scalars
# scaler = torch.amp.GradScaler()

os.environ['CUDA_VISIBLE_DEVICES']= '0'
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

# device = ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')
num_epochs =10
batch_size = 2
val_batch_size = 20
shuffle = True
pin_memory = True
num_workers = 1
rootpath=r''
frame_interval=1
input_shape=[224,224]
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_data=DADA(rootpath , 'training', interval=1,transform=transform)
val_data=DADA(rootpath , 'testing', interval=1,transform=transform)

traindata_loader=DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,drop_last=True)

valdata_loader=DataLoader(dataset=val_data, batch_size=val_batch_size , shuffle=False,
                                  num_workers=num_workers, pin_memory=True,drop_last=True)


def write_scalars(logger, epoch, loss):
    logger.add_scalars('train/loss',{'loss':loss}, epoch)

def write_test_scalars(logger, epoch, losses, metrics):
    # logger.add_scalars('test/loss',{'loss':loss}, epoch)
    logger.add_scalars('test/losses/total_loss',{'Loss': losses}, epoch)
    logger.add_scalars('test/accuracy/AP',{'AP':metrics['AP'], 'PR80':metrics['PR80']}, epoch)
    logger.add_scalars('test/accuracy/time-to-accident',{'mTTA':metrics['mTTA'], 'TTA_R80':metrics['TTA_R80']}, epoch)

def train():
    # the path to save model
    model_dir ='model_crossvlt_tiny_ep_10_b2_no_swin_weights'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logs_dir = 'train_crossvlt_tiny_ep_10_b2_no_swin_weights'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger = SummaryWriter(logs_dir)
    h_dim = 256
    n_layers = 1
    depth=4
    adim=opt.adim
    heads=opt.heads
    num_tokens=opt.num_tokens
    c_dim=opt.c_dim
    s_dim1=opt.s_dim1
    s_dim2=opt.s_dim2
    keral=opt.keral
    num_class=opt.num_class
    # model = AccidentXai(num_classes, x_dim, h_dim, z_dim,n_layers).to(device)
    model=improved_accident(h_dim,n_layers,depth,adim,heads,num_tokens,c_dim,s_dim1,s_dim2,keral,num_class).to(device)

    #Add CrossVLT parameters to the optimizer
    cross_vlt = model.cross_vlt

    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in cross_vlt.backbone.named_parameters():
        if "layers.blocks" not in name:
            if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                backbone_no_decay.append(m)
            else:
                backbone_decay.append(m)
    
    opt1 = torch.optim.Adam([

        #CrossVLT's parameters
        {'params': backbone_no_decay, 'weight_decay': 0.0, "lr": 3e-5},
        {'params': backbone_decay, "lr": 3e-5},
        {"params": [p for p in cross_vlt.lang_stage1.encoder.parameters() if p.requires_grad], "lr": 3e-5},
        {"params": [p for p in cross_vlt.lang_stage2.parameters() if p.requires_grad], "lr": 3e-5},
        {"params": [p for p in cross_vlt.lang_stage3.parameters() if p.requires_grad], "lr": 3e-5},
        {"params": [p for p in cross_vlt.lang_stage4.parameters() if p.requires_grad], "lr": 3e-5},
        {"params": [p for p in cross_vlt.downsample_2.parameters() if p.requires_grad], "lr": 1e-3},
        
        #Uncomment when using Swin Base
        #{"params": [p for p in cross_vlt.downsample_1.parameters() if p.requires_grad], "lr": 1e-3},

        {'params': model.features.parameters(), 'lr': 1e-6},
        {'params': model.deconv.parameters(), 'lr': 1e-4},
        {'params': model.gru_net.parameters(), 'lr': 1e-5},
    ])

    scheduler1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=1, gamma=0.1)

    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(traindata_loader ,total = len(traindata_loader), leave = True)
        for imgs, focus, info, label, texts in loop:
            labels=label
            toa = info[0:, 4].to(device)
            loop.set_description(f"Epoch  [{epoch+1}/{num_epochs}]")
            # print(imgs.shape)
            imgs = imgs.to(device)
            focus = focus.to(device)
            labels= np.array(labels).astype(int)
            labels = torch.from_numpy(labels)
            labels =labels.to(device)
            model.to(device)

            # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            total_loss, outputs = model(imgs ,focus,labels.long(),toa,texts)

            # scaler.scale(loss['total_loss'].mean()).backward()
            # scaler.unscale_(opt1)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            opt1.step()
            opt1.zero_grad()


            # scaler.step(opt1)
            # scaler.update()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss = total_loss)

        write_scalars(logger,epoch,total_loss)

        if (epoch+1) % 5 == 0:
            scheduler1.step()
        #test and evaluate the model
        if (epoch+1) % 1==0:
            model.eval()
            ap, mTTA_0_5, mTTA, tta_r80, auc, total_loss = validation(valdata_loader, model)
            metrics = {"AP": ap, "mTTA_0_5": mTTA_0_5, "mTTA": mTTA, "TTA_R80": tta_r80, "AUC": auc, "avg_val_loss": total_loss}
            write_validation_scalars(logger, epoch, metrics)
            model.train()
            model_file = os.path.join(model_dir, 'saved_model_%02d.pth'%(epoch))
            torch.save(model.state_dict(),model_file)
        logger.close()

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    train()
