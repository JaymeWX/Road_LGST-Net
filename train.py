import torch
import torch.utils.data as data
import os
import warnings
from time import time
from networks.unet import Unet
from networks.dunet import Dunet
from networks.linknet import LinkNet34
from networks.dlinknet import DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from networks.nllinknet import NL34_LinkNet
from networks.windvitnet import ViTRoad
from networks.SETR import SETR
from networks.Unet3plus import UNet_3Plus
from networks.SegNet import SegNet

from framework import MyFrame
from loss import dice_bce_loss
from data import DeepGlobeDataset, RoadDataset
from networks.SWATNet import build_road_SWATNet
import csv
warnings.filterwarnings("ignore")



def get_deepglobe_trainset():
    ROOT = 'dataset/deepglobe/train/'
    val_data_csv = 'dataset/deepglobe/train.csv'
    with open(val_data_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        vallist = list(map(lambda x: x[0], reader))
    # imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(ROOT))
    # trainlist = list(map(lambda x: x[:-8], imagelist))
    
    dataset = DeepGlobeDataset(vallist, ROOT, is_train=True, img_size=img_size)
    return dataset

def get_roadtrace_trainset():
    ROOT = 'dataset/roadtracer_mydata/1024x1024/'
    val_data_txt = 'dataset/roadtracer_mydata/train_list_1024.txt'
    with open(val_data_txt, 'r') as f:
        vallist = [name.replace('\n', '') for name in f.readlines()]
    
    dataset = RoadDataset(vallist, ROOT, is_train=True, img_size=img_size)
    return dataset

def get_massroad_trainset():
    ROOT = 'dataset/Massachusetts_roads/tiff_750x750/all/'
    val_data_txt = 'dataset/Massachusetts_roads/tiff_750x750/train.txt'
    with open(val_data_txt, 'r') as f:
        vallist = [name.replace('\n', '') for name in f.readlines()]
    
    dataset = RoadDataset(vallist, ROOT, is_train=True, img_size=img_size)
    return dataset

def train(model_name, dataset_method, img_size, batch_size, log_name, checkpoint = ''):
    # net = ViTRoad(image_size = img_size, patch_size=16, dim=256, depth = 4, heads = 6, dim_head = 64, mlp_dim = 512)
    # net = ViTRoad()
    
    dataset_name = str(dataset_method.__name__).split('_')[1]
    NAME = f'{model_name}_{dataset_name}_{log_name}'
    total_epoch = 3000  # 训练轮数
    # net = build_road_sam(192, 6, img_size=img_size, encoder_depth = 12, decoder_depth = 12) #, checkpoint='weights/ImageEncoderViT-192.pt')

    # net = SETR(num_classes=1, image_size=512, patch_size=512//16, dim=1024, depth = 24, heads = 16, mlp_dim = 2048, out_indices = (9, 14, 19, 23))
    # net = ViTRoad(img_size)
    # net = UNet_3Plus()
    if model_name == 'SETR':
        net = SETR(num_classes=1, image_size=512, patch_size=512//16, dim=1024, depth = 24, heads = 16, mlp_dim = 2048, out_indices = (9, 14, 19, 23))
    elif model_name == 'SWATNet':
        net = build_road_SWATNet(192, 6, img_size=img_size, encoder_depth = 12, decoder_depth = 12)
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


    solver = MyFrame(net, dice_bce_loss, lr = 4e-6, lr_end= 1e-6, epochs = total_epoch)
    if checkpoint is not '':
        solver.load(checkpoint)
    dataset = dataset_method()
    # dataset = get_massroad_trainset()
    # dataset = get_roadtrace_trainset()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)
    mylog = open(f'logs/{NAME}.log', 'w')
    # tic = time()
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
        train_epoch_loss /= len(data_loader_iter)
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # checkpoint = 'weights/dlinknet.pt'
    batchsize = 8
    img_size = 512
    train('DLinkNet', get_deepglobe_trainset, img_size = img_size, 
          log_name='v1', batch_size=batchsize)
