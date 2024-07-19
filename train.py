import torch
import os
import warnings
from time import time
from networks.unet import Unet
from networks.dunet import Dunet
from networks.linknet import LinkNet34
from networks.dlinknet import DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from networks.nllinknet import NL34_LinkNet
from networks.SETR import SETR
from networks.Unet3plus import UNet_3Plus
from networks.SegNet import SegNet
from networks.SWATNet import build_road_SWATNet
from networks.SWATNet_v2 import SWATNet as SWATNetV2
from networks.SWATNet_dlink import DinkNet34 as SWATNetV3

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

def get_massroad_trainset(img_size):
    ROOT = 'dataset/Massachusetts_roads/tiff_750x750/all/'
    val_data_txt = 'dataset/Massachusetts_roads/tiff_750x750/train.txt'
    with open(val_data_txt, 'r') as f:
        vallist = [name.replace('\n', '') for name in f.readlines()]
    
    dataset = RoadDataset(vallist, ROOT, is_train=True, img_size=img_size)
    return dataset

def train(model_name, dataset_method, img_size, batch_size, log_name, checkpoint = ''):
    
    dataset_name = str(dataset_method.__name__).split('_')[1]
    NAME = f'{model_name}_{dataset_name}_{log_name}'
    total_epoch = 300  

    if model_name == 'SETR':
        net = SETR(num_classes=1, image_size=512, patch_size=512//16, dim=1024, depth = 24, heads = 16, mlp_dim = 2048, out_indices = (9, 14, 19, 23))
    elif model_name == 'SWATNet':
        net = build_road_SWATNet(192, 6, img_size=img_size, encoder_depth = 6, decoder_depth = 6)
    elif model_name == 'SWATNetV2':
        net = SWATNetV2(192, 6, img_size=img_size, encoder_depth = 6, decoder_depth = 6)
    elif model_name == 'SWATNetV3':
        net = SWATNetV3()
    elif model_name == 'NoSWATNet':
        net = build_road_SWATNet(192, 6, img_size=img_size, encoder_depth = 12, decoder_depth = 12, is_SWAT = False)
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

    solver = ModelContainer(net, dice_bce_loss, lr = 5e-4, lr_end= 1e-6, epochs = total_epoch)
    if checkpoint is not '':
        solver.load(checkpoint)
    dataset = dataset_method(img_size)
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
        print(f'epoch:[{epoch}/{total_epoch}] loss:{train_epoch_loss:.6f} lr:{lr:.8f} time:{int(time() - tic)/60:.2f}min', file = mylog)

        
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
    # checkpoint = 'weights/dlinknet.pt'
    batchsize = 16
    img_size = 512
    # SWATNet NoSWATNet SETR DLinkNet NLLinkNet SWATNetV3
    train('SWATNetV3', get_deepglobe_trainset, img_size = img_size, 
          log_name='v2', batch_size=batchsize, checkpoint='weights/SWATNetV3_deepglobe_v1.pt')
