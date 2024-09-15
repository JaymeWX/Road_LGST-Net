import torch
import os
import warnings
from time import time
from networks.unet import Unet
from networks.Unet3plus import UNet_3Plus
from networks.SETR import SETR
from networks.SegNet import SegNet
from networks.dlinknet import DinkNet34
from networks.nllinknet import NL34_LinkNet
from networks.SWATNet import SWATNet


from framework import ModelContainer
from loss import dice_bce_loss
from data import DeepGlobeDataset, RoadDataset
import csv
warnings.filterwarnings("ignore")

def get_deepglobe_trainset(img_size, data_num = -1):
    ROOT = 'dataset/deepglobe/train/'
    train_data_csv = 'dataset/deepglobe/train.csv'
    with open(train_data_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        trainlist = list(map(lambda x: x[0], reader))

    trainlist = trainlist[:data_num]
    dataset = DeepGlobeDataset(trainlist, ROOT, is_train=True, img_size=img_size)
    return dataset

def get_roadtrace_trainset(img_size):
    ROOT = 'dataset/roadtracer_mydata/1024x1024/'
    val_data_txt = 'dataset/roadtracer_mydata/train_list_1024.txt'
    with open(val_data_txt, 'r') as f:
        vallist = [name.replace('\n', '') for name in f.readlines()]
    
    dataset = RoadDataset(vallist, ROOT, is_train=True, img_size=img_size)
    return dataset

def train(model_name, dataset_method, img_size, batch_size, log_name, checkpoint = '', lr = 1e-4, lr_end = 1e-6, total_epoch = 300):
    
    dataset_name = str(dataset_method.__name__).split('_')[1]
    NAME = f'{model_name}_{dataset_name}_{log_name}' 

    if model_name == 'SETR':
        net = SETR(num_classes=1, image_size=512, patch_size=512//16, dim=1024, depth = 24, heads = 16, mlp_dim = 2048, out_indices = (9, 14, 19, 23))
    elif model_name == 'SWATNet':
        net = SWATNet(scale='normal')
    elif model_name == 'DLinkNet':
        net = DinkNet34()
    elif model_name == 'NLLinkNet':
        net = NL34_LinkNet()
    elif model_name == 'UNet':
        net = Unet()
    elif model_name == 'UNet3Plus':
        net = UNet_3Plus()
    elif model_name == 'SegNet':
        net = SegNet()
    else:
        print('invaild model name!!')
        return

    solver = ModelContainer(net, dice_bce_loss, lr = lr, lr_end= lr_end, epochs = total_epoch)
    if checkpoint is not '':
        solver.load(checkpoint)
    dataset = dataset_method(img_size)
    train_img_count = len(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)
    mylog = open(f'logs/{NAME}.log', 'w')
    no_optim = 0
    train_epoch_best_loss = 100.

    print('start train')
    for epoch in range(1, total_epoch + 1):
        tic = time()
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        batch_num = len(data_loader_iter)
        for index, (img, mask, names) in enumerate(data_loader_iter):
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
            lr = solver.optimizer.param_groups[0]["lr"]
            print(f'epoch:[{epoch}/{total_epoch}] batch:[{index}/{batch_num}] loss:{train_loss:.6f} lr:{lr:.8f}  {NAME}')
        train_epoch_loss /= batch_num
        print(f'epoch:[{epoch}/{total_epoch}] loss:{train_epoch_loss:.6f} lr:{lr:.8f} time:{int(time() - tic)/60:.4f}min fps:{train_img_count/int(time() - tic):.2f}', file = mylog)

        
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('weights/' + NAME + '.pt')
        
        solver.update_lr_by_scheduler(epoch)
        mylog.flush()

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()

if __name__ == '__main__':
    #config 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    batchsize = 16
    img_size = 512   
    lr, lr_end = 1e-4, 1e-7
    total_epoch = 600
    
    # SWATNet SETR DLinkNet NLLinkNet SWATNetV3 UNet
    train('SWATNet', get_roadtrace_trainset, img_size = img_size, 
          log_name='debug_v1', batch_size=batchsize, checkpoint='', lr=lr, lr_end=lr_end, total_epoch=total_epoch)
