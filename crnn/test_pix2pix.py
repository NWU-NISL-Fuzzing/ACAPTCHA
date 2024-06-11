'''
测试准确率，并删除错误的图像
'''

import os
import torch
from torch.autograd import Variable
from tqdm import *
import utils
import dataset
from PIL import Image
from network import crnn_cnn7 as crnn


import params
import argparse
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_data_from_folder(folder_path):
    pics = os.listdir(folder_path)
    # pBar = tqdm(range(0, len(pics)))
    total = len(pics)
    correct = 0
    wrong = 0

    for pic in pics:
        predictLabel = val(folder_path + '/' + pic)
        # pBar.update(1)

        if (pic.__contains__('-')):
            trueLable = pic.split('-')[0]
        else:
            trueLable = pic.split('_')[0]

        if trueLable == predictLabel:
            correct += 1
            # print(trueLable + "被预测为" + predictLabel + " " + pic)



        else:
            wrong += 1
            # print(trueLable + "被预测为" + predictLabel + " " + pic)
            # os.remove(folder_path + '/' + pic)



            # rm_path = "dataset/test_class/7r"
            # #
            # name = os.listdir(rm_path)
            # for i in name:
            #     path = rm_path + '/{}'.format(i)
            #     if trueLable in i:
            #         os.remove(path)


        # os.remove(folder_path + '/' + pic)

    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('错误个数为' + str(wrong))


# net init


def val(image_path):
    nclass = len(params.alphabet) + 1
    model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
    if torch.cuda.is_available():
        model = model.cuda()

    # load model
    # print('loading pretrained model from %s' % model_path)
    if params.multi_gpu:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


    converter = utils.strLabelConverter(params.alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(image_path).convert('RGB')
    # image = Image.open(image_path).convert('L')
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

    file_name_list = ['v7']
    model_name_list = ['cnn7', 'alexnet', 'vgg16']
    alp = 'i-fgsm_'
    # eps_list = ['0.06/', '0.08/', '0.1/']
    eps_list = ['0.06_', '0.08_', '0.1_']
    die_list = ['10', '25', '50']
    for file_name in file_name_list:
        for model_name in model_name_list:
            for eps in eps_list:
                for die in die_list:

                    print(str(file_name) + "/" + model_name + "/" + alp + eps + str(die))

                    model_path = './pix2pix_models/v78_pix2pix.pth'
                    # model_path = './pix2pix_models/v9_pix2pix.pth'
                    # model_path = './pix2pix_models/v10_pix2pix.pth'

                    folder_path = "F:/bishe/pix2pix/results/" + str(file_name) + "/" + model_name + "/" + alp + eps + str(die) + "/resize/"
                    read_data_from_folder(folder_path)



