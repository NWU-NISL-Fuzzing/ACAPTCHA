'''
统计去噪结果
'''



import os
import torch
from torch.autograd import Variable
from tqdm import *
import utils
import dataset
from PIL import Image
# from network import crnn_cnn7 as crnn
from network import crnn_alexnet as crnn
# from network import crnn_vgg16 as crnn
import params
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 测试滤波情况

def read_data_from_folder(folder_path):
    pics = os.listdir(folder_path)
    pBar = tqdm(range(0, len(pics)))
    total = len(pics)
    correct = 0
    wrong = 0

    static = {'bit_depth_4': 0,
              'bit_depth_5': 0,
              'bit_depth_6': 0,
              'bit_depth_7': 0,
              'bilater': 0,
              'guassian': 0,
              'mean': 0,
              'median': 0,
              # 'compression_png': 0,
              'jpeg_compression_75': 0,
              'resizing': 0,
              'tvloss': 0,
              "generated": 0}

    for pic in pics:

        predictLabel = val(folder_path + '/' + pic)

        pBar.update(1)

        if (pic.__contains__('-')):
            trueLable = pic.split('-')[0]
        else:
            trueLable = pic.split('_')[0]

        # print(trueLable + "被预测为" + predictLabel)

        if trueLable == predictLabel:
            correct += 1
            # copy(folder_path + '/' + pic, folder_path + '_true/' + pic)

            # os.remove(folder_path + '/' + pic)
            # os.remove(folder_path_adv + '/' + pic)

            # copy(folder_path_org + '/' + pic, pix_org + '/' + pic)
            # copy(folder_path + '/' + pic, pix_adv + '/' + pic)

            if '4_bit_depth' in pic:
                static['bit_depth_4'] += 1
            if '5_bit_depth' in pic:
                static['bit_depth_5'] += 1
            if '6_bit_depth' in pic:
                static['bit_depth_6'] += 1
            if '7_bit_depth' in pic:
                static['bit_depth_7'] += 1
            if 'bilater' in pic:
                static['bilater'] += 1
            if 'Guassian' in pic:
                static['guassian'] += 1
            if 'mean' in pic:
                static['mean'] += 1
            if 'median' in pic:
                static['median'] += 1
            # if 'compression.png' in pic:
            #     static['compression_png'] += 1
            if 'jpeg_compression_75' in pic:
                static['jpeg_compression_75'] += 1
            if 'resizing' in pic:
                static['resizing'] += 1
            if 'tvloss' in pic:
                static['tvloss'] += 1

            if 'generated' in pic:
                static['generated'] += 1

            # copy(folder_path + '/' + pic, pix_adv + '/' + pic)

            # rm_path = '../datasets/500/v3/pix2pix/org_imgs'
            # name = os.listdir(rm_path)
            # for i in name:
            #     path = rm_path + '/{}'.format(i)
            #     if trueLable in i:
            #         # print(i)
            #         os.remove(path)
            # # print(pic)
            # os.remove(folder_path + '/' + pic)


        else:
            wrong += 1
            # rm_path = '../datasets/500/v3/pix2pix/org_imgs'
            # name = os.listdir(rm_path)
            # for i in name:
            #     path = rm_path + '/{}'.format(i)
            #     if trueLable in i:
                    # print(i)
                    # os.remove(path)
            # print(pic)
            # os.remove(folder_path + '/' + pic)
            # os.remove(folder_path_adv + '/' + pic)


            # rm_path = "dataset/test_class/7r"
            # #


        # os.remove(folder_path + '/' + pic)

    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('错误个数为' + str(wrong))

    static = {k: (v / 500)*100 for k, v in static.items()}
    print(static)


# net init


def val(image_path):
    nclass = len(params.alphabet) + 1
    model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)

    if torch.cuda.is_available():
        model = model.cuda()

    # load model
    # print('loading pretrained model from %s' % model_path)
    # if params.multi_gpu:
    #     model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    converter = utils.strLabelConverter(params.alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(image_path)
    image = transformer(image)

    if torch.cuda.is_available():
        image = image.cuda()

    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()

    preds = model(image)

    _, preds = preds.max(2)

    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model_path', type=str, help='crnn model path', default='F:/bishe/crnn/models/cnn7_v9.pth')
    parser.add_argument('-m', '--model_path', type=str, help='crnn model path', default='F:/bishe/crnn/models/alexnet_v13.pth')
    # parser.add_argument('-m', '--model_path', type=str, help='crnn model path', default='F:/bishe/crnn/models/vgg16_v7.pth')
    parser.add_argument('-f', '--folder_path', type=str, help='test folder path', default='F:/bishe/datasets/version/v13/alexnet/filter/fgsm/0.08/')
    args = parser.parse_args()
    model_path = args.model_path
    folder_path = args.folder_path
    read_data_from_folder(folder_path)