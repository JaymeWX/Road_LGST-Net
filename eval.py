import numpy as np
import torch
import warnings
from framework import MyFrame
from loss import dice_bce_loss
from networks.dlinknet import DinkNet34
from networks.nllinknet import NL34_LinkNet
from networks.unet import Unet
from networks.SETR import SETR
from networks.Unet3plus import UNet_3Plus
from networks.SegNet import SegNet
from data import DeepGlobeDataset, RoadDataset
from torch.utils.data import DataLoader
import csv
from PIL import Image
import os
from networks.sam import build_road_sam, Attention
from fvcore.nn import FlopCountAnalysis, parameter_count_table

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
        # self.hist = self.fast_hist(label, pred, 2)
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

    def get_IOU(self):
        iou = (self.Iand + self.smooth_factor) / (self.Ior + self.smooth_factor)
        return iou
    
def get_deepglobe_testset(img_size):
    ROOT = 'dataset/deepglobe/train/'
    val_data_csv = 'dataset/deepglobe/val.csv'
    with open(val_data_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        vallist = list(map(lambda x: x[0], reader))
    
    dataset = DeepGlobeDataset(vallist, ROOT, is_train=False, img_size = img_size)
    return dataset

def get_roadtrace_testset(img_size):
    ROOT = 'dataset/roadtracer_mydata/1024x1024/'
    val_data_txt = 'dataset/roadtracer_mydata/test_list_1024.txt'
    with open(val_data_txt, 'r') as f:
        vallist = [name.replace('\n', '') for name in f.readlines()]
    
    dataset = RoadDataset(vallist, ROOT, is_train=False, img_size = img_size)
    return dataset

def get_massroad_testset(img_size):
    ROOT = 'dataset/Massachusetts_roads/tiff_750x750/all/'
    val_data_txt = 'dataset/Massachusetts_roads/tiff_750x750/test.txt'
    with open(val_data_txt, 'r') as f:
        vallist = [name.replace('\n', '') for name in f.readlines()]
    
    dataset = RoadDataset(vallist, ROOT, is_train=False, img_size = img_size)
    return dataset

def metrics_eval(model_name, dataset_method, model_weigth, img_size = 512, save_output_mask = True):
    labels = []
    predicts = []
    # img_path = "eval_example/104_sat.jpg"
    # label_path = "eval_example/104_mask.png"
    model_name = model_name
    dataset_name = str(dataset_method.__name__).split('_')[1]
    save_path = f'results/{model_name}_{dataset_name}/'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    img_size = 512
    
    if model_name == 'SETR':
        net = SETR(num_classes=1, image_size=512, patch_size=512//16, dim=1024, depth = 24, heads = 16, mlp_dim = 2048, out_indices = (9, 14, 19, 23))
    elif model_name == 'SWATNet':
        net = build_road_sam(192, 6, img_size=img_size, encoder_depth = 12, decoder_depth = 12)
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
    
    # 加载模型
    solver = MyFrame(net, dice_bce_loss, 2e-4)
    # solver.load("weights/trainlog_WindVitNet_v1.pt")
    solver.load(f"weights/{model_weigth}.pt")
    
    # 加载数据集
    batchsize = 8
    # dataset = get_deepglobe_testset(img_size)
    dataset = dataset_method(img_size)
    # dataset = get_massroad_testset(img_size)

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

    precisions = []
    recalls = []
    accuracies = []
    ious = []
    for pre_mask, label in zip(predicts, labels):
        accuracy = AccuracyIndex(label, pre_mask)
        precisions.append(accuracy.get_precision())
        recalls.append(accuracy.get_recall())
        accuracies.append(accuracy.get_accuracy())
        ious.append(accuracy.get_IOU())
    prec, recall, acc, iou = list(map(lambda x: sum(x)/len(x), [precisions, recalls, accuracies, ious]))
    print(f'precision:{prec} recall:{recall} accuracy:{acc} iou:{iou}')

    # 评估
    el = IOUMetric()
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)
    
    print('acc: ', acc)
    print('acc_cls: ', acc_cls)
    print('iou: ', iou)
    print('miou: ', miou)
    print('fwavacc: ', fwavacc)


def param_gflops_eval(model_name, img_size = 512):
    
    # pyprof.init()
    # profiler.start()
    # patch_row_num = 32
    # dim = 192
    if model_name == 'SETR':
        net = SETR(num_classes=1, image_size=512, patch_size=512//16, dim=1024, depth = 24, heads = 16, mlp_dim = 2048, out_indices = (9, 14, 19, 23))
    elif model_name == 'SWATNet':
        net = build_road_sam(192, 6, img_size=img_size, encoder_depth = 12, decoder_depth = 12)
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
    net = net.cuda()
    input = torch.randn(1, 3, img_size, img_size).cuda()  

    #评估模型参数量和计算量              
    # flops, params = profile(net, inputs=(input, ))      
    # print(flops,params) 
    # flops,params = clever_format([flops, params],"%.3f")
    # print(f'flops: {flops} , params: {params}')  

    # 分析FLOPs
    flops = FlopCountAnalysis(net, input)
    print("FLOPs: ", flops.total()/1000000000.0)
    params = parameter_count_table(net, max_depth=1)
    print(params)

if __name__ == '__main__':
    # param_gflops_eval('SWATNet')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    metrics_eval('UNet', get_deepglobe_testset, 'UNet_deepglobe_v1')