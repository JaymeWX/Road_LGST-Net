
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import warnings
from framework import ModelContainer
from loss import dice_bce_loss
from networks.dlinknet import DinkNet34
from networks.nllinknet import NL34_LinkNet
from networks.unet import Unet
from networks.SETR import SETR
from networks.Unet3plus import UNet_3Plus
from networks.SegNet import SegNet
from networks.LGSTNet import LGSTNet
from data import DeepGlobeDataset, RoadDataset
from torch.utils.data import DataLoader
import csv
from PIL import Image
import torch
warnings.filterwarnings("ignore")


class IOUMetric:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, fwavacc

class AccuracyIndex():
    def __init__(self, label:np.array, pred:np.array) -> None:
        self.Iand = np.sum(label*pred) 
        self.Ior = np.sum(label) + np.sum(pred) - self.Iand
        self.pix_count = label.shape[-1] * label.shape[-2] 
        self.label_count = np.sum(label)
        self.pred_count = np.sum(pred)
        self.smooth_factor = 1e-10

    def get_accuracy(self):
        acc = (self.pix_count - self.Ior + self.Iand)/self.pix_count
        return acc
    
    def get_precision(self):
        pre = (self.Iand + self.smooth_factor)/(self.pred_count + self.smooth_factor)
        return pre
    
    def get_recall(self):
        rec = (self.Iand + self.smooth_factor)/(self.label_count + self.smooth_factor)
        return rec
    
def get_deepglobe_testset(img_size):
    ROOT = 'dataset/deepglobe/train/'
    val_data_csv = 'dataset/deepglobe/val.csv'
    with open(val_data_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        vallist = list(map(lambda x: x[0], reader))
    
    dataset = DeepGlobeDataset(vallist, ROOT, is_train=False, img_size = img_size)
    return dataset

def get_roadtrace_testset(img_size):
    ROOT = '../Road_SWATNet/dataset/roadtracer_mydata/1024x1024/'
    val_data_txt = '../Road_SWATNet/dataset/roadtracer_mydata/test_list_1024.txt'
    
    with open(val_data_txt, 'r') as f:
        vallist = [name.replace('\n', '') for name in f.readlines()]
    
    dataset = RoadDataset(vallist, ROOT, is_train=False, img_size = img_size)
    return dataset

def net_init(model_name, img_size = 512):
    if model_name == 'SETR':
        net = SETR(num_classes=1, image_size=img_size, patch_size=img_size//16, dim=1024, depth = 24, heads = 16, mlp_dim = 2048, out_indices = (9, 14, 19, 23))
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
    elif model_name == 'LGSTNet':
        net = LGSTNet()
        pre_data = torch.randn(1, 3, img_size, img_size)
        # to initialize weights which created in the running time
        net(pre_data)
    else:
        print('invaild model name!!')
        return None
    return net

def metrics_eval(model_name, dataset_method, model_weigth, img_size = 512, save_output_mask = True):
    labels = []
    predicts = []
    model_name = model_name
    dataset_name = str(dataset_method.__name__).split('_')[1]
    save_path = f'results/{model_name}_{dataset_name}/'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    img_size = 512
    
    net = net_init(model_name, img_size)
    # 加载模型
    solver = ModelContainer(net, dice_bce_loss, 2e-4)
    solver.load(f"weights/{model_weigth}")
    
    # 加载数据集
    batchsize = 8
    dataset = dataset_method(img_size)

    data_loader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0)
    
    batch_num = len(data_loader)
    for index, (img_batch, mask_batch, image_names) in enumerate(data_loader):
        solver.set_input(img_batch)
        mask_pre, _ = solver.test_batch()

        for ids, mask_p in enumerate(mask_pre):
            temp = np.array(mask_p, np.int64)
            label = mask_batch[ids]
            label = label.cpu().data.numpy().squeeze(0)
            predicts.append(temp)
            labels.append(label)
            img = temp * 255
            img = np.array(img, np.uint8)
            pil_image = Image.fromarray(img)
            if save_output_mask:
                pil_image.save(save_path+f'{image_names[ids]}.png', 'PNG')
        print(f'progress: {index}/{batch_num}')


    # 评估
    precisions = []
    recalls = []
    accuracies = []
    for pre_mask, label in zip(predicts, labels):
        accuracy = AccuracyIndex(label, pre_mask)
        precisions.append(accuracy.get_precision())
        recalls.append(accuracy.get_recall())
        accuracies.append(accuracy.get_accuracy())
    prec, recall, acc = list(map(lambda x: sum(x)/len(x), [precisions, recalls, accuracies]))
    F1_ = 2*recall*prec/(recall+prec)
    print(f'recall:{recall} precision:{prec} F1:{F1_} accuracy:{acc}')
    
    el = IOUMetric()
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)
    
    print('acc: ', acc)
    print('acc_cls: ', acc_cls)
    print('iou: ', iou)
    print('miou: ', miou)
    print('fwavacc: ', fwavacc)



if __name__ == '__main__':
    metrics_eval('LGSTNet', get_roadtrace_testset, 'LGSTNet_roadtrace_debug_v1.pt')