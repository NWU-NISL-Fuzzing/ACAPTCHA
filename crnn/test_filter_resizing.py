'''
统计去噪结果
'''



import os
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
# from network import crnn_cnn7 as crnn
# from network import crnn_alexnet as crnn
from network import crnn_vgg16 as crnn
import params


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 测试滤波情况

def read_data_from_folder(folder_path):
    pics = os.listdir(folder_path)

    total = len(pics)
    correct = 0
    wrong = 0
    static = {'resizing': 0}

    for pic in pics:

        predictLabel = val(folder_path + '/' + pic)


        if (pic.__contains__('-')):
            trueLable = pic.split('-')[0]
        else:
            trueLable = pic.split('_')[0]

        # print(trueLable + "被预测为" + predictLabel)

        if trueLable == predictLabel:
            correct += 1
            if 'resizing' in pic:
                static['resizing'] += 1

        else:
            wrong += 1


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


    file_name_list = ['v7', 'v8', 'v9', 'v10']
    model_name = 'vgg16'
    alp_list = ['fgsm/', 'deepfool/', 'i-fgsm/']
    for alp in alp_list:
        if alp == 'fgsm/':
            eps_list = ['0.06/', '0.08/', '0.1/']
            for file_name in file_name_list:
                for eps in eps_list:
                    print(str(file_name) + "/" + model_name + alp + eps)
                    model_path = './models/' + model_name + '_' + file_name + '.pth'
                    folder_path = "F:/bishe/datasets/version/" + file_name + "/" + model_name + "/resizing/" + alp + eps
                    read_data_from_folder(folder_path)

        elif alp == 'deepfool/':
            eps_list = ['1/', '5/', '10/', '50/']
            for file_name in file_name_list:
                for eps in eps_list:
                    print(str(file_name) + "/" + model_name + alp + eps)
                    model_path = './models/' + model_name + '_' + file_name + '.pth'
                    folder_path = "F:/bishe/datasets/version/" + file_name + "/" + model_name + "/resizing/" + alp + eps
                    read_data_from_folder(folder_path)

        elif alp == 'i-fgsm/':
            eps_list = ['0.06/', '0.08/', '0.1/']
            die_list = ['10/', '25/', '50/']
            for file_name in file_name_list:
                for eps in eps_list:
                    for die in die_list:
                        print(str(file_name) + "/" + model_name + alp + eps + die)
                        model_path = './models/' + model_name + '_' + file_name + '.pth'
                        folder_path = "F:/bishe/datasets/version/" + file_name + "/" + model_name + "/resizing/" + alp + eps + die
                        read_data_from_folder(folder_path)
