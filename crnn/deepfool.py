import os
import time
import torch
from PIL import Image
from torch import optim
from torch.autograd import Variable
from torch.nn import CTCLoss
from torchvision import transforms
# from network import crnn_cnn7 as net
from network import crnn_alexnet as net
# from network import crnn_vgg16 as net
import params, utils, dataset
import copy
import numpy as np
from torch.autograd.gradcheck import zero_gradients

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def net_init():
    nclass = len(params.alphabet) + 1
    crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
    crnn.apply(weights_init)
    # crnn.load_state_dict(torch.load(("models/cnn7_v9.pth"), map_location=torch.device('cpu')))
    crnn.load_state_dict(torch.load(("models/alexnet_v7_2.pth"), map_location=torch.device('cpu')))
    # crnn.load_state_dict(torch.load(("models/vgg16_v9.pth"), map_location=torch.device('cpu')))
    return crnn


crnn = net_init()

# -----------------------------------------------
"""
In this block
    Init some utils defined in utils.py
"""
# Compute average for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.averager()

# Convert between str and label.
converter = utils.strLabelConverter(params.alphabet)

"""
In this block
    criterion define
"""
criterion = CTCLoss()

"""
In this block
    Init some tensor
    Put tensor and net on cuda
    NOTE:
        image, text, length is used by both val and train
        becaues train and val will never use it at the same time.
"""

image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

    crnn = crnn.cuda()
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(params.ngpu))

image = Variable(image)
text = Variable(text)
length = Variable(length)

# -----------------------------------------------
"""
In this block
    Setup optimizer
"""
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

# -----------------------------------------------

criterion = CTCLoss(zero_infinity=True)



def deepfool(image, net, num_classes=10, max_iter=5, overshoot=0.02):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02). 防止梯度消失
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    # 获取计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 原label 第一个字母的label
    orig_label = np.argmax(net(image).data.cpu().numpy())
    print("label={}".format(orig_label))

    # 图像数据梯度可以获取
    image.requires_grad = True

    output = net(image)  # 前向传播
    input_shape = image.cpu().detach().numpy().shape  # detach 将Variable从graph中分离

    w = np.zeros(input_shape)
    r_rot = np.zeros(input_shape)  # 更新矩阵

    for epoch in range(max_iter):

        scores = net(image).data.cpu().numpy()[0]
        scores = scores.flatten()
        label = np.argmax(scores)
        print("epoch={} label={} score={}".format(epoch, label, scores[label]))

        # 如果无定向攻击成功
        if label != orig_label:
            break

        pert = np.inf  # 用来记录最小距离
        output = torch.squeeze(output, dim=1)
        output[0, orig_label].backward(retain_graph=True)  # 只管当前类别的情况
        grad_orig = image.grad.data.cpu().numpy().copy()  # 前一次梯度

        for k in range(1, num_classes):
            if k == orig_label:
                continue

            # 梯度清零
            zero_gradients(image)

            output[0, k].backward(retain_graph=True)
            cur_grad = image.grad.data.cpu().numpy().copy()

            # 计算w_k和f_k
            w_k = cur_grad - grad_orig
            f_k = (output[0, k] - output[0, orig_label]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())  # 距离
            # 选择pert最小值，提速作用，快速收敛到最易攻破的类别
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # 计算r_i和r_tot
        r_i = (pert + 1e-8) * w / np.linalg.norm(w)
        r_tot = np.float32(r_rot + r_i)

        image.data = image.data + (1 + overshoot) * torch.from_numpy(r_tot).to(device)

    adv_img = image

    return adv_img


def save_image(tensor, label, epoch):

    # path = '../datasets/version/v9/cnn7/deepfool/' + str(max_iter)
    path = '../datasets/version/v7/alexnet_2/deepfool/' + str(max_iter)
    # path = '../datasets/version/v9/vgg16/deepfool/' + str(max_iter)
    # path = '../datasets/version/v10/pix2pix/train/bg/deepfool/' + str(max_iter)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension

    unloader = transforms.ToPILImage()
    image = unloader(image)

    # image = image.resize((100, 32))

    # transformer = dataset.resizeNormalize((160, 60))
    # image = transformer(image)
    #
    # image = unloader(image)

    time_stamp = str(int(time.time()))
    filename = label + '_' + time_stamp + '_' + str(max_iter) + '.png'

    if not os.path.exists(path):
        os.makedirs(path)
    # filename = label + '.png'

    save_path = path + "/" + filename
    # print(save_path)
    # time.sleep(1)
    image.save(save_path)


def val(image_path, true_label):
    # imageC = Image.open(image_path).convert('RGB')
    # transformer = dataset.resizeNormalize((100, 32))
    # imageC = transformer(imageC)
    image = Image.open(image_path).convert('RGB')
    transformer = dataset.resizeNormalize((100, 32))
    image = transformer(image)

    image = image.view(1, *image.size())
    batch_size = image.size(0)

    if (true_label.__contains__('-')):
        trueLable = true_label.split('-')[0]
    # if (true_label.__contains__('.')):
    #     trueLable = true_label.split('.')[0]
    else:
        trueLable = true_label.split('_')[0]

    LableVerble = bytes(trueLable, encoding="utf8")
    LableVerble = (LableVerble,)

    text, length = converter.encode(LableVerble)

    image.requires_grad = True

    preds = crnn(image)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size

    cost.backward()

    loss_avg.add(cost)

    # data_grad = image.grad.data

    perturbed_data = deepfool(image, crnn, len(params.alphabet) + 1, max_iter)

    # perturbed_data = fgsm_attack(image, epsilon, data_grad)
    # perturbed_data = fgsm_attack(image, epsilon, data_grad)
    save_image(perturbed_data, trueLable, max_iter)


def read_data_from_folder(folder_path):
    pics = os.listdir(folder_path)
    for pic in pics:
        print(pic)
        val(folder_path + '/' + pic, pic)
        # print("----------------------------", folder_path + '/' + pic)


if __name__ == '__main__':
    # epsilon = float(input("please input epsilon = "))
    crnn = net_init()

    max_iter_list = [1, 50]
    for max_iter in max_iter_list :
        # read_data_from_folder("../datasets/version/v9/cnn7/org_img")
        read_data_from_folder("../datasets/version/v7/alexnet_2/org_img")
        # read_data_from_folder("../datasets/version/v9/vgg16/org_img")
        # read_data_from_folder("../datasets/version/v10/pix2pix/train/bg/1")

