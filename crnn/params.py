# import alphabets




alphabet = """1
2
3
4
5
6
7
8
9
0
Q
W
E
R
T
Y
U
I
O
P
A
S
D
F
G
H
J
K
L
Z
X
C
V
B
N
M
q
w
e
r
t
y
u
i
o
p
a
s
d
f
g
h
j
k
l
z
x
c
v
b
n
m
"""

# about data and net
# alphabet = alphabets.alphabet
keep_ratio = False # whether to keep ratio for image resize
manualSeed = 1234 # reproduce experiemnt
random_sample = True # whether to sample the dataset with random sampler
imgH = 32 # the height of the input image to network
# imgH = 60 # the height of the input image to network
imgW = 100 # the width of the input image to network
# imgW = 160 # the width of the input image to network
nh = 256 # size of the lstm hidden state
nc = 3  # 输入通道数

pretrained = '/home/nisl2/nislcopy1/che/captcha/system/crnn/expr/CRNN_189_2.pth' # path to pretrained model (to continue training)
# pretrained = '' # path to pretrained model (to continue training)
expr_dir = 'expr' # where to store samples and models
dealwith_lossnan = False # whether to replace all nan/inf in gradients to zero

# hardware
cuda = True # enables cuda
multi_gpu = False # whether to use multi gpu
ngpu = 3 # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 0 # number of data loading workers

# training process
displayInterval = 2500 # interval to be print the train loss
valInterval = 1720 # interval to val the model loss and accuray
saveInterval = 1720 # interval to save model
n_val_disp = 10 # number of samples to display when val the model

# finetune
nepoch = 200 # number of epochs to train for
batchSize = 64 # input batch size
lr = 0.00001 # learning rate for Critic, not used by adadealta
beta1 = 0.5 # beta1 for adam. default=0.5
adam = False # whether to use adam (default is rmsprop)
adadelta = False # whether to use adadelta (default is rmsprop)
