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

"""
fgsm_attack
"""


def _where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()

    return (cond * x) + ((1 - cond) * y)

# i-FGSM attack code、
def i_fgsm_attack(image, epsilon, text, length, batch_size, alpha=1, iteration=1):
    image_org = image

    for j in range(iteration):

        preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size

        cost.backward()

        loss_avg.add(cost)

        data_grad = image.grad.data
        sign_data_grad = data_grad.sign()

        image = image + alpha * sign_data_grad

        image = _where(image > image_org + epsilon, image_org + epsilon, image)

        image = _where(image < image_org - epsilon, image_org - epsilon, image)

        image = torch.clamp(image, 0, 1)

        image = Variable(image.data, requires_grad=True)

        # save_image1(image, trueLable, j, epsilon)

    return image

def save_image(tensor, label, epsilon, iteration):
    # path = '../datasets/version/v9/cnn7/i-fgsm/' + str(epsilon) + '/' + str(iteration)
    path = '../datasets/version/v7/alexnet_2/i-fgsm/' + str(epsilon) + '/' + str(iteration)
    # path = '../datasets/version/v9/vgg16/i-fgsm/' + str(epsilon) + '/' + str(iteration)

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
    filename = label + '_' + time_stamp + '_' + str(epsilon) + '.png'

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
    alpha = 1
    # i_fgsm_attack(image, epsilon, text, length, trueLable, batch_size, alpha=1, iteration=10)
    perturbed_data = i_fgsm_attack(image, epsilon, text, length, batch_size, alpha, iteration)
    # perturbed_data = fgsm_attack(imageC, epsilon, data_grad)
    save_image(perturbed_data, trueLable, epsilon, iteration)


def read_data_from_folder(folder_path):
    pics = os.listdir(folder_path)
    for pic in pics:
        print(pic)
        val(folder_path + '/' + pic, pic)
        # print("----------------------------", folder_path + '/' + pic)


if __name__ == '__main__':
    # epsilon = float(input("please input epsilon = "))
    crnn = net_init()
    eps_list = [0.06, 0.08, 0.1]
    iteration_list = [10, 25, 50]


    for iteration in iteration_list:
        for epsilon in eps_list:
            # read_data_from_folder("../datasets/version/v9/cnn7/org_img")
            read_data_from_folder("../datasets/version/v7/alexnet_2/org_img")
            # read_data_from_folder("../datasets/version/v9/vgg16/org_img")

