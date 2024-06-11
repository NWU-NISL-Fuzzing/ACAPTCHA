# -*- coding: utf-8 -*-
"""
生成有背景的验证码用于训练一个模型识别pix2pix去噪后的图像准确率
"""

# -*- coding: utf-8 -*-
"""
1. 生成验证码（9种都齐全）


加背景（bg）,防止去背景扰动
加干扰线 随机3-4根
collapse，有字符重叠的情况

字符长度随机 4-9个字符
字体大小可变 15-25
多字体 10种，每张验证码中每个字符随机
筛选字符集,没有 1-I-l o-O-0 2-Z-z S-s V-v X-x w-W u-U
rotate -30~-20 没有warp之前朝一个方向
Warp




"""

import os
import random
import time

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from generate_captcha_tools import image_rotate, image_warp, random_color, sin_warp_x

WIDTH = 100
HEIGHT = 32

# 多字体
def get_ttf_list(folder):
    """
    返回文件加中所有的图片对象的iamge
    """
    ttf_list = []
    for file in os.listdir(folder):
        ttf_path = os.path.join(folder, file)
        ttf_list.append(ttf_path)
    return ttf_list



def get_rotate_offset(rotate, image):
    """
    因图片旋转而产生的误差
    """
    # print("w = %d ,off = %d "%(image_w,abs(rotate)))
    # image_w, _ = image.size
    # tmp_a = abs((math.cos(abs(rotate) * math.pi / 180) * image_w))
    # tmp_b = abs((math.sin(abs(rotate) * math.pi / 180) * image_w))
    # return (int)(tmp_a + tmp_b - image_w)
    return abs(rotate)



def create_noise_line(image,width,line_num,font_color):
    """
    sohu的干扰线
    :param image: 需要添加的图片
    :param width: 线条最大的宽度
    :param line_num: 线条的个数
    :return: image
    """
    def create_line(image, width):
        (w,h) = image.size
        x_1 = random.randint(0,int(w/3))
        x_2 = random.randint(int(w/3)*2,w)
        y_1 = random.randint(0,h)
        y_2 = random.randint(0,h)
        draw = ImageDraw.Draw(image)
        # line_color = random_color(0, 180)
        draw.line(((x_1, y_1), (x_2, y_2)), fill=font_color, width=width)
        return image
    for _ in range(line_num):
        image = create_line(image,random.randint(0, width))
    return image


def generate_captcha(code, font_color, bg_image, ttf_list, captcha_path
                     ):
    """
    生成验证码
    :param code: 字符内容
    :param font_color: 字体颜色
    :param background: 背景颜色
    :param captcha_path: 保存路径
    :return: null
    """

    tmp_falg = 5
    # FONT_SIZE = random.randint(33, 38)
    (b_WIDTH, b_HEIGHT, b_FONT_SIZE) = (int(WIDTH * tmp_falg), int(HEIGHT * tmp_falg),
                                        int(FONT_SIZE * tmp_falg))
    # # 背景图片
    # bg_image = bg_image.resize((WIDTH * tmp_falg, HEIGHT * tmp_falg),
    #                            Image.ANTIALIAS)
    # image_bg = bg_image.copy().convert('RGBA')

    image_bg = Image.new('RGBA', (b_WIDTH, b_HEIGHT), bg_image)
    draw_avatar = ImageDraw.Draw(image_bg)

    # 画干扰线
    # tmp_flag1 = int(b_HEIGHT / 2 - b_HEIGHT / 10)
    # x_1 = random.randint(0 - int(b_WIDTH / 10), int(b_WIDTH / 10))
    # if x_1 < 0:
    #     x_1 = 0
    # y_1 = random.randint(0 + tmp_flag1, b_HEIGHT - tmp_flag1)
    # x_2 = random.randint(b_WIDTH - int(b_WIDTH / 10),
    #                      b_WIDTH + int(b_WIDTH / 10))
    # if x_2 > b_WIDTH:
    #     x_2 = b_WIDTH
    # y_2 = random.randint(0 + tmp_flag1, b_HEIGHT - tmp_flag1)
    #
    # draw_avatar.line((x_1, y_1, x_2, y_2), font_color, width=15)
    #
    # amplitude = random.randint(0, int(b_HEIGHT))  # 振幅
    # period = random.randint(int(b_WIDTH / 2), int(b_WIDTH * 2))  # 周期
    # phase = random.randint(0, period)  # 相位
    # image_bg = sin_warp_x(image_bg, amplitude, period, phase, font_color)


    images = []  # 存放字符
    myrotate = random.randint(-30, -20)
    # myrotate = 0
    rotate_offset_list = []  # 因为旋转再粘贴的时候产生的误差

    def _draw_character(char):
        """
        根据 char 生成image
        """

        font = ImageFont.truetype(random.choice(ttf_list), b_FONT_SIZE)  # 获取字体
        char_w, char_h = draw_avatar.textsize(char, font=font)  # 得到字符的长宽

        im_char = Image.new('RGBA', (char_w, char_h))
        draw_avatar_char = ImageDraw.Draw(im_char)

        draw_avatar_char.text((0, 0), char, fill=font_color, font=font)
        del draw_avatar_char  # 贴上去之后删除单个图片
        pre_rotate = im_char.copy()
        # rotate
        im_char = image_rotate(im_char, -30, -20)
        # im_char = 0
        # warp
        im_char = image_warp(pre_rotate, im_char, [0.8, 1.0], [1.0, 1.2])

        return im_char

    for char in code:
        image = _draw_character(char)
        rotate_offset_list.append(get_rotate_offset(myrotate, image))
        images.append(image)
    # print(myrotate)
    # start = random.randint(0, int(b_WIDTH / 8))  # 开始粘贴的水平位置
    start = int(b_WIDTH / 15)  # 开始粘贴的水平位置
    # start = 15  # 开始粘贴的水平位置
    # start = 0  # 开始粘贴的水平位置
    # step = -6  # 两个字符之间的距离
    step = 0  # 两个字符之间的距离

    # 估算一些总长度是否越界
    sum_width = start
    for i, image in enumerate(images):
        char_w, _ = image.size
        sum_width = sum_width + char_w
        if i == len(images) - 1:
            sum_width = sum_width
        else:
            sum_width = sum_width + step - rotate_offset_list[i]

    if sum_width > b_WIDTH:
        image_bg = image_bg.resize((sum_width, b_HEIGHT), Image.ANTIALIAS)

    offset = start
    # y_offset = random.randint(-int(b_HEIGHT / 8), int(b_HEIGHT / 8))

    # y_offset = - int(b_HEIGHT / 8)
    y_offset = 0
    # fill_color = (255, 193, 193)

    # 干扰线
    # ImageDraw.Draw(image_bg).line([(0, b_HEIGHT), (b_WIDTH / 3, 0)], fill=(120, 120, 120), width=15)
    # ImageDraw.Draw(image_bg).line([(b_WIDTH / 3, b_HEIGHT), (2 * b_WIDTH / 3, 0)], fill=(120, 120, 120), width=15)
    # ImageDraw.Draw(image_bg).line([(2 * b_WIDTH / 3, b_HEIGHT), (b_WIDTH, 0)], fill=(120, 120, 120), width=15)

    # ImageDraw.Draw(image_bg).line([(0, b_HEIGHT), (b_WIDTH/3, 0)], fill=(120, 120, 120), width=15)
    #
    # ImageDraw.Draw(image_bg).line([(b_WIDTH/3, b_HEIGHT), (2 * b_WIDTH / 3, 0)], fill=(120, 120, 120), width=15)
    # ImageDraw.Draw(image_bg).line([(2 * b_WIDTH / 3, b_HEIGHT), (b_WIDTH, 0)], fill=(120, 120, 120), width=15)
    # ImageDraw.Draw(image_bg).line([(0, b_HEIGHT), (b_WIDTH/3, 0)], fill=(0, 0, 0), width=15)
    # ImageDraw.Draw(image_bg).line([(b_WIDTH/3, b_HEIGHT), (2 * b_WIDTH / 3, 0)], fill=(0, 0, 0), width=15)
    # ImageDraw.Draw(image_bg).line([(2 * b_WIDTH / 3, b_HEIGHT), (b_WIDTH, 0)], fill=(0, 0, 0), width=15)

    for i, image in enumerate(images):
        char_w, char_h = image.size
        mask = image
        # y_offset_2 = random.randint(-int(b_HEIGHT / 12), int(b_HEIGHT / 12))
        y_offset_2 = 0

        image_bg.paste(image, (offset, int(
            (b_HEIGHT - char_h) / 2) + y_offset + y_offset_2), mask)
        # step_random = random.randint(-10, 10)
        step_random = 2
        offset = offset + char_w + step - rotate_offset_list[i] + step_random

    # 波浪化
    # if flag:
    # amplitude = random.randint(int(WIDTH / 5), int(WIDTH / 5))  # 振幅
    # period = random.randint(int(b_HEIGHT), int(b_HEIGHT))  # 周期
    # phase = random.randint(0, 0)  # 相位
    # image_bg = sin_warp_y(image_bg, amplitude, period, phase, background)
    # image_bg2 = sin_warp_y(image_bg2, amplitude, period, phase, background)

    # image_bg = image_bg.resize((WIDTH, HEIGHT), Image.ANTIALIAS).convert("L")

    # #干扰线
    # width = 7 # 对大的宽度
    # line_num = random.randint(3, 4)
    # # line_num = 3
    # image_bg = create_noise_line(image_bg, width, line_num,font_color)
    #
    # image_bg = image_bg.resize((WIDTH, HEIGHT), Image.ANTIALIAS)

    image_bg.save(captcha_path)


def random_color_in_list():
    # rgb1 = (156, 3, 137)
    # rgb2 = (156, 3, 137)
    # rgb3 = (156, 3, 137)
    # rgb4 = (5, 22, 97)
    # rgb5 = (11, 59, 179)
    # rgb6 = (88, 85, 20)
    # rgb7 = (19, 15, 143)
    # rgb8 = (144, 28, 33)
    # rgb9 = (109, 96, 165)
    # rgb10 = (66, 48, 149)
    # rgb11 = (165, 31, 116)
    # rgb12 = (16, 4, 175)
    # rgb13 = (79, 31, 26)
    # rgb14 = (129, 28, 97)
    # rgb15 = (129, 28, 97)
    # rgb16 = (103, 57, 175)

    list = [(156, 3, 137), (5, 22, 97), (11, 59, 179), (88, 85, 20), (19, 15, 143),
            (144, 28, 33), (109, 96, 165), (66, 48, 149), (165, 31, 116), (16, 4, 175), (79, 31, 26), (129, 28, 97),
            (129, 28, 97), (103, 57, 175)]

    rgb = random.choice(list)

    return rgb


def get_bg_list(folder):
    """
    返回文件加中所有的图片对象的iamge
    """
    bg_list = []
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        bg_list.append(Image.open(image_path))
    return bg_list


def createUnderThreshold():
# def createUnderThreshold(threshold):
    """
    在区间之间产生颜色
    :param start: 开始像素区间
    :param end: 终止像素区间
    :param opacity: 透明度
    :return: (r,g,b)
    """
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    # # 考虑字符颜色限制
    # while (red * 0.299 + green * 0.587 + blue * 0.114 > threshold):
    #     red = random.randint(0, 255)
    #     green = random.randint(0, 255)
    #     blue = random.randint(0, 255)

    return red, green, blue


def main(captcha_text, captcha_save_path):
    labels_list = captcha_text.strip().split("#")
    pBar = tqdm(total=len(labels_list))
    background = (255, 255, 255)  # 验证码背景颜色

    # 循环生成
    for i, each in enumerate(labels_list):
        pBar.update(1)
        now = str(int(time.time()))
        captcha_path = os.path.join(captcha_save_path, each + '--_' + str(now) + '.png')
        flag = False
        # if i % (random.randint(1, 10)) == 0:        #     flag = True
        # fill_color = random_color(0, 180)
        #       $colorLevel = $r * 0.299 + $g * 0.587 + $b * 0.114;
        # fill_color = random_color(0, 180)
        # fill_color = random_color_in_list()
        # fill_color = (80,80,80)
        # fill_color = (0, 0, 0)

        fill_color = createUnderThreshold()

        # fill_color = (255,193,193)

        # 选择肉眼友好的颜色

        # fill_color = (80,80,80)
        # fill_color = (60,60,60)
        # fill_color = (200,200,200)warp

        # 图片背景
        bg_image_folder = 'bg/bg'
        bg_list = get_bg_list(bg_image_folder)
        ttf_list = get_ttf_list('TTF_V9')

        # generate_captcha(each, fill_color, random.choice(bg_list), random.choice(ttf_list), captcha_path)
        generate_captcha(each, fill_color, background, ttf_list, captcha_path)

        # create_save_4captcha_image(each,captcha_path)


def random_captcha_text(char_set, captcha_size):
    """
    返回一个验证码
    :param char_set:  字符集合
    :param captcha_size: 每个验证码的长度
    :return: 一个组合好的验证码list
    """
    captcha_text = []
    for _ in range(captcha_size):
        char = random.choice(char_set)
        captcha_text.append(char)
    return captcha_text


def gen_captcha_text(char_set, captcha_size):
    """
    返回一个验证码String
    :param char_set:  字符集合
    :param captcha_size: 每个验证码的长度
    :return: 一个组合好的验证码list
    """
    captcha_text = random_captcha_text(char_set, captcha_size)  # 返回的是list
    captcha_text = ''.join(captcha_text)  # 返回的是str
    return captcha_text


def generate_labels(char_set, captcha_number, captcha_size):
    """
    :param char_set: 字符集合
    :param captcha_number: 验证码的长度
    :param captcha_size:  每个验证码的长度
    :return: 0
    """
    captcha_list = [
        gen_captcha_text(char_set, captcha_size) for _ in range(captcha_number)
    ]
    captcha_text = '#'.join(captcha_list)
    return captcha_text


def save_labels(captcha_text, save_path):
    """
    :param captcha_text:  captcha String
    :param save_path:  保存位置
    :return: 0
    """
    f = open(save_path, 'w')
    f.write(captcha_text)
    f.close()


def make_dir(folder):
    """
    判断文件夹是不是存在,不存在则创建
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        pass


def generate_labels(char_set, captcha_number, captcha_size):
    """
    :param char_set: 字符集合
    :param captcha_number: 验证码的长度
    :param captcha_size:  每个验证码的长度
    :return: 0
    """
    captcha_list = [
        gen_captcha_text(char_set, captcha_size) for _ in range(captcha_number)
    ]
    captcha_text = '#'.join(captcha_list)
    return captcha_text


if __name__ == '__main__':
    i = 0
    # 重复随机次数
    while i < 10000:
        # 设置字符集，没有 1-I-l o-O-0 2-Z-z S-s V-v X-x w-W u-U
        char_set = [
            '3', '4', '5', '6', '7', '8', '9',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm',
            'n', 'p', 'q', 'r', 't', 'y',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
            'N', 'P', 'Q', 'R', 'T', 'Y'

            # 'o', '0', 'O', 'S', 's', 'U', 'V', 'W', 'X', 'Z',
            # 'I', 'l', '1', 'u', 'v', 'w', 'x', 'z'
        ]

        # 随机生成一个字符长度
        captcha_size = random.randint(4, 9)

        # 随机字体大小
        FONT_SIZE = random.randint(15, 25)

        # 每次在该长度下生成的数据集数量
        captcha_number_train = 10
        captcha_number_val = 1
        captcha_number_test = 1
        captcha_number_500 = 1

        captcha_text_train = generate_labels(char_set, captcha_number_train, captcha_size)
        captcha_text_val = generate_labels(char_set, captcha_number_val, captcha_size)
        captcha_text_test = generate_labels(char_set, captcha_number_test, captcha_size)
        captcha_text_500 = generate_labels(char_set, captcha_number_500, captcha_size)

        captcha_save_path_root = '../datasets/version/V9_pix2pix1/'
        captcha_save_path_500_root = '../datasets/500/v9_pix2pix1/org_imgs/'
        # 空白背景的图像
        # captcha_save_path_500_root1 = '../datasets/500/v4/pix2pix/train/org_imgs_white/'

        captcha_save_path_train = captcha_save_path_root + 'train'
        captcha_save_path_val = captcha_save_path_root + 'val'
        captcha_save_path_test = captcha_save_path_root + 'test'
        captcha_save_path_500 = captcha_save_path_500_root

        make_dir(captcha_save_path_train)
        make_dir(captcha_save_path_val)
        make_dir(captcha_save_path_test)
        make_dir(captcha_save_path_500)

        main(captcha_text_train, captcha_save_path_train)
        main(captcha_text_val, captcha_save_path_val)
        main(captcha_text_test, captcha_save_path_test)
        main(captcha_text_500, captcha_save_path_500)

        i += 1
