import numpy as np
import matplotlib.pyplot as plt
import re
def read_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
    # test = re.split(r'[ :]', data[0])
    # data = [re.split(r'[ :]', item) for item in data]
    losses = []
    for item in data:
        item_ls = re.split(r'[ :]', item)
        if len(item_ls) > 6:
            losses.append(float(item_ls[3]))
    
    return losses
    
def loss_draw():
    losses1 = read_data('logs/DLinkNet_roadtrace__v1.log')
    losses2 = read_data('logs/NLLinkNet_roadtrace__v1.log')
    losses3 = read_data('logs/SegNet_roadtrace__v1.log')
    losses4 = read_data('logs/SWATNetV6_roadtrace_ablation_v4.log')
    losses5 = read_data('logs/UNet_roadtrace__v1.log')
    losses6 = read_data('logs/UNet3Plus_roadtrace__v1.log')
    losses7 = read_data('logs/SETR_roadtrace__v1.log')
    
    
    # 绘制损失曲线图
    plt.plot(range(len(losses6)), losses6, '-', lw=1, label="UNet3Plus")
    plt.plot(range(len(losses5)), losses5, '--', lw=1, label="UNet")
    plt.plot(range(len(losses3)), losses3, ':', lw=1, label="SegNet")
    plt.plot(range(len(losses7)), losses7, '-', lw=1, label="SETR")
    plt.plot(range(len(losses1)), losses1, '--', lw=1, label="DLinkNet")
    plt.plot(range(len(losses2)), losses2, ':', lw=1, label="NLLinkNet")
    plt.plot(range(len(losses4)), losses4, '-', lw=1, label="SWATNet")
    
    plt.xlabel("Epoch", fontsize = 14)
    plt.ylabel("Loss", fontsize = 14)
    # plt.title("Irregular Loss Curve")
    plt.legend()
    plt.savefig('loss.png')
    plt.savefig('loss.pdf')

def computational_draw():
    img_size = [256,384,512,640,768,896,1024]
    img_size = [str(num) for num in img_size]
    attention_mb = [1853,4157,8619,17819,31711,49000]
    swat_mb = [1953,3767,6137,9753,12855,17581,22145]
    attention_mb = [i/1000 for i in attention_mb]
    swat_mb = [i/1000 for i in swat_mb]
    plt.plot(img_size, swat_mb, marker = 'o', lw=1, label="ViT")
    plt.plot(img_size[:len(attention_mb)], attention_mb, marker = '^', lw=1, label="SWAT")
    
    plt.xlabel("Image Size", fontsize = 14)
    plt.ylabel("GPU Memory (GB)", fontsize = 14)
    # plt.title("GPU Memory Cur")
    plt.legend()
    plt.savefig('computational.pdf')


def draw_memory():

# % Unet3+ 44770
# % Unet 4288
# % SegNet 24660
# % SETR  18184
# % D-LinkNet 8288
# % NL-LinkNe 8664
# % SWATNet-S 10310
# % SWATNet-L 12784
    categories = ['Unet3+', 'Unet', 'SegNet', 'SETR', 'D-LinkNet', 'NL-LinkNet', 'SWATNet-S', 'SWATNet-L']
    values1 = [44.770, 4.288, 24.660, 18.184, 8.288, 8.664, 10.310, 14.594]
    # values2 = [6, 8, 10]
    categories.reverse()
    values1.reverse()
    bar_width = 0.75
    
    plt.figure(figsize=(10,4))

    # plt.grid()
    plt.barh(categories, values1, bar_width)

    
    for x, y in zip(range(len(categories)), values1):
        plt.text(y+1, x-0.1, y, fontsize = 10)
    plt.xlim(0, 50)
    plt.xlabel('Memory cost (GB)', fontsize = 14)
    plt.ylabel('Comparison Methods', fontsize = 14)
    # plt.title('Horizontal Bar Chart')
    plt.tight_layout()
    plt.savefig('memory.pdf')


if __name__ == '__main__':
    computational_draw()
    loss_draw()
    draw_memory()
    # 1、‘-’：实线样式
    # 2、‘--’：短横线样式
    # 3、‘-.’：点划线样式
    # 4、‘:’：虚线样式
    # 5、‘.’：点标记
    # 6、‘o’：圆标记
    # 7、‘V’：倒三角标记
    # 8、‘^’：正三角标记
    # 9、‘<’：左三角标记
    # 10、‘>’：右三角表示
    # 11、‘1’：下箭头标记
    # 12、‘2’：上箭头标记
    # 13、‘3’：左箭头标记
    # 14、‘4’：右箭头标记
    #     颜色样式： 

    # 1、‘b’ ：蓝色
    # 2、‘c’： 青绿色
    # 3、‘g’： 绿色
    # 4、‘k’ ：黑色
    # 5、‘m’：洋红色
    # 6、‘r’： 红色
    # 7、‘w’：白色
    # 8、‘y’： 黄色
    