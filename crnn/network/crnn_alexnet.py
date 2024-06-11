import torch.nn as nn
import torch.nn.functional as F



class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3]  # kernel_size
        ss = [1, 1, 1, 1, 1]  # stride
        ps = [1, 1, 1, 1, 1]  # padding
        nm = [64, 192, 384, 384, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # 32*100
        convRelu(0)  # 32*100
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2, 0))  # 16*50

        convRelu(1)   # 16*50
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((4, 4), (4, 2), (0, 1)))  # 4*25

        convRelu(2, True)  # 4*25
        convRelu(3)  # 4*25
        convRelu(4)  # 4*25
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((4, 3), (4, 1), (0, 1)))  # 1*25



        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))


    def forward(self, input):
        # conv features
        conv = self.cnn(input)


        b, c, h, w = conv.size()
        # print(b, c, h, w )

        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]


        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero


