import os

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from tqdm import *
import cv2
import numpy
import numpy as np
from PIL import Image
import argparse


def reszing_shrink(img_path, scaling):
    img = cv2.imread(img_path)
    height, width = img.shape[0:2]

    # 缩小图像
    shrink_size = (int(width / scaling), int(height / scaling))
    shrink = cv2.resize(img, shrink_size)

    return shrink


def reszing_enlarge(img, scaling, true_label):
    # img = cv2.imread(img_path)
    height, width = img.shape[0:2]

    # 放大图像
    enlarge_size = (int(width * scaling), int(height * scaling))
    enlarge = cv2.resize(img, enlarge_size, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(filter_folder + true_label + "_resizing.png", enlarge)


def compression(img_path, true_label):
    img = cv2.imread(img_path)
    img_copy = img.copy()

    cv2.imwrite(filter_folder + '/' + true_label + "_jpeg_compression_75.jpg", img_copy,
                [int(cv2.IMWRITE_JPEG_QUALITY), 75])  # 0-100压缩质量，越高质量越好，默认95.论文中设成75
    # cv2.imwrite(filter_folder + '/' + true_label + "_png_compression.png", img_copy,
    #             [int(cv2.IMWRITE_PNG_COMPRESSION), 9])  # 0-9压缩级别，较高的值意味着更小的尺寸和更长的压缩时间而默认值是3


def bit_depth_reduction(img, bit):
    img = numpy.array(img)  # 变成numpy形式
    img = img / 255  # 值限制到[0, 1]
    img = np.rint(img * (np.power(2, bit) - 1))  # bit-depth从8位变至4位，四舍五入取整
    img = img / np.power(2, bit) - 1  # 值限制到[0, 1]
    img = img * 255  # 等比例恢复至8bit的像素值
    img = Image.fromarray(np.uint8(img))
    # img.save(save_path + file_name + ".png")
    return img


def bit_depth_reduction_filtering_test(org_img, trueLable, bit):
    img_bit = bit_depth_reduction(org_img, bit)  # pil
    img_bit.save(filter_folder + '/' + trueLable + '_' + str(bit) + '_bit_depth_reduction.png')


def filtering_once(img):
    ########     四个不同的滤波器    #########

    img = numpy.array(img)  # PIL -> Numpy

    # 均值滤波
    img_mean = cv2.blur(img, (3, 3))
    # matplotlib.image.imsave('./filtering_per_img/n02129604/out_30_img_mean.png', img_mean)

    # 高斯滤波
    img_Guassian = cv2.GaussianBlur(img, (3, 3), 0)
    # matplotlib.image.imsave('./filtering_per_img/n02129604/out_30_img_Guassian.png', img_Guassian)

    # 中值滤波
    img_median = cv2.medianBlur(img, 3)
    # matplotlib.image.imsave('./filtering_per_img/n02129604/out_30_img_median.png', img_median)

    # 双边滤波
    img_bilater = cv2.bilateralFilter(img, 3, 75, 75)
    # matplotlib.image.imsave('./filtering_per_img/n02129604/out_30_img_bilater.png', img_bilater)

    img_mean = transforms.ToPILImage()(img_mean)
    img_Guassian = transforms.ToPILImage()(img_Guassian)
    img_median = transforms.ToPILImage()(img_median)
    img_bilater = transforms.ToPILImage()(img_bilater)

    return img_mean, img_Guassian, img_median, img_bilater


def filtering_once_test(org_img, true_label):
    img_mean, img_Guassian, img_median, img_bilater = filtering_once(org_img)  # pil

    img_mean.save(filter_folder + '/' + true_label + '_img_mean.png')
    img_Guassian.save(filter_folder + '/' + true_label + '_img_Guassian.png')
    img_median.save(filter_folder + '/' + true_label + '_img_median.png')
    img_bilater.save(filter_folder + '/' + true_label + '_img_bilater.png')


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def tvloss(path, true_label):
    addition = TVLoss()  # .to(device)
    #    z = addition(x)
    #    print(x)
    #    print(z.data)
    #    z.backward()
    #    print(x.grad)
    img = cv2.imread(path)
    # img_t=torch.from_numpy(img)
    a, b, c = img.shape
    img_t = img
    img_v = Variable(torch.FloatTensor(img_t).view(1, a, b, c), requires_grad=True)  # .to(device)
    # print(img_v)
    img_z = addition(img_v)  # .to(device)
    img_z.backward()
    # img = img_v.numpy()
    # images = torch.clamp(images.data + t * x_grad, 0, 1)

    cv2.imwrite(filter_folder + '/' + true_label + "_tvloss.png", img)


def resizing(path, true_label):
    img = cv2.imread(path)
    img_resizing_big = cv2.resize(img, (0, 0), fx=2, fy=2)
    img_resizing = cv2.resize(img_resizing_big, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite(filter_folder + '/' + true_label + "_resizing.png", img_resizing)


def imagequilting(path, trueLable):
    IT = cv2.imread(path)
    # IT.dtype=np.double
    ITm, ITn, ITk = IT.shape
    # IT=torch.from_numpy(IT)
    Block = 20
    BlockO = 2
    PTm = 60
    PTn = 160
    PT = np.zeros((PTm + Block, PTn + Block, ITk), np.double)
    # PT=torch.from_numpy(PT)
    for l in range(Block, PTm + Block, Block - BlockO):
        for m in range(Block, PTn + Block, Block - BlockO):
            if l == Block and m == Block:
                PT[0:Block, 0:Block, :] = IT[0:Block, 0:Block, :]
            else:
                if l == Block:
                    B1 = PT[0:l, m - Block:m - Block + BlockO, :]
                    B2 = ME1(B1, Block, BlockO, ITm, ITn, IT)
                    PT[0:l, m - Block:m, :] = B2
                else:
                    if m == Block:
                        B3 = PT[l - Block:l + BlockO - Block, 0:m, :]
                        B4 = ME3(B3, Block, BlockO, ITm, ITn, IT)
                        PT[l - Block:l, 0:m, :] = B4
                    else:
                        B5 = PT[l - Block:l, m - Block:m + BlockO - Block, :]
                        # print(B5.shape)
                        B6 = PT[l - Block:l + BlockO - Block, m - Block:m, :]
                        # print(B6.shape)
                        B7 = ME2(B5, B6, Block, BlockO, ITm, ITn, IT)
                        PT[l - Block:l, m - Block:m, :] = B7
    PTout = PT[0:PTm, 0:PTn, :]
    # PTout=PTout.numpy()
    cv2.imwrite(filter_folder + trueLable + "_imageQuilting.png", PTout)


def read_data_from_folder(folder_path):
    pics = os.listdir(folder_path)
    pBar = tqdm(range(0, len(pics)))

    for pic in pics:
        # predictLabel = val(folder_path + '/' + pic)
        pBar.update(1)
        if (pic.__contains__('-')):
            trueLable = pic.split('-')[0]
        else:
            trueLable = pic.split('_')[0]

        org_img = get_image(folder_path + '/' + pic)

        # 4滤波
        filtering_once_test(org_img, trueLable)

        # 位深滤波
        for bit in range(4, 8):
            bit_depth_reduction_filtering_test(org_img, trueLable, bit)

        # 压缩画质
        compression(folder_path + '/' + pic, trueLable)
        # scaling = 2
        # reszing_enlarge(reszing_shrink(folder_path + '/' + pic, scaling),scaling,trueLable)

        # tvloss
        tvloss(folder_path + '/' + pic, trueLable)

        # resizing
        resizing(folder_path + '/' + pic, trueLable)

        # imagequilting(folder_path + '/' + pic,trueLable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder_path', type=str, help='test folder path', default='v13')
    parser.add_argument('-model_name', '--model_name', type=str, help='test folder path', default='alexnet')
    parser.add_argument('-adv_pert', '--adv_pert', type=str, help='test folder path', default='fgsm')
    parser.add_argument('-eps', '--eps', type=str, help='test folder path', default='0.08')
    parser.add_argument('-iteration', '--iteration', type=str, help='test folder path', default='50')
    args = parser.parse_args()
    org_folder = args.folder_path

    # # fgsm
    folder_path = '../datasets/version/' + org_folder + '/' + args.model_name + '/' + args.adv_pert + '/'+ args.eps
    filter_folder = "../datasets/version/" + org_folder + '/' + args.model_name  + '/filter/' + args.adv_pert + '/'+ args.eps


    # # i-fgsm
    # folder_path = '../datasets/version/' + org_folder + '/' + args.model_name + '/' + args.adv_pert + '/' + args.eps + '/' + args.iteration
    # filter_folder = "../datasets/version/" + org_folder + '/' + args.model_name + '/filter/' + args.adv_pert + '/' + args.eps + '/' + args.iteration

    # deepfool
    # folder_path = '../datasets/version/' + org_folder + '/' + args.model_name + '/' + args.adv_pert + '/' + args.iteration
    # filter_folder = "../datasets/version/" + org_folder + '/' + args.model_name + '/filter/' + args.adv_pert + '/' + args.iteration

    if not os.path.exists(filter_folder):  # 如果路径不存在
        os.makedirs(filter_folder)

    read_data_from_folder(folder_path)
