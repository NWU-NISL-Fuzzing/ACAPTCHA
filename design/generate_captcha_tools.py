# -*- coding: utf-8 -*-
"""
1. 生成验证码的工具类
"""
__author__ = 'big_centaur'

import math
import random
from PIL import Image, ImageDraw


def image_rotate(im_char, start, end):
    """
    旋转函数
    :param im_char: image类型的图片
    :param start: 区间开始角度
    :param end: 区间终止角度
    :return: 旋转完之后的图片 PIL.image格式
    """
    im_char = im_char.crop(im_char.getbbox())
    im_char = im_char.rotate(
        random.uniform(start, end), Image.BILINEAR, expand=1)
    return im_char


def image_warp(pre_rotate, im_char, list1=(0.1, 0.3), list2=(0.2, 0.4)):
    """
    扭曲函数 开始两个对象都是image 第一个是为了去没有旋转之前的尺寸 若没有旋转操作可和第二个参数一样
    :param pre_rotate:
    :param im_char:
    :param list1: 扭曲的程度 [0.1 ,0.3]
    :param list2: 扭曲的程度 [0.2 ,0.4]
    :return:
    """
    # im_char.show()
    if list1 is None:
        list1 = [0.1, 0.3]
    (w, h) = im_char.size
    dx = w * random.uniform(list1[0], list1[1])
    dy = h * random.uniform(list2[0], list2[1])
    x1 = int(random.uniform(-dx, dx))
    y1 = int(random.uniform(-dy, dy))
    x2 = int(random.uniform(-dx, dx))
    y2 = int(random.uniform(-dy, dy))
    w2 = w + abs(x1) + abs(x2)
    h2 = h + abs(y1) + abs(y2)
    # 变量data是一个8元组(x0, y0, x1, y1, x2, y2, x3, y3)，它包括源四边形的左上，左下，右下和右上四个角。
    data = (
        x1, y1,
        -x1, h2 - y2,
        w2 + x2, h2 + y2,
        w2 - x2, -y1,
    )
    im_char = im_char.resize((w2, h2))
    # im_char.show()
    im_char = im_char.transform((w, h), Image.QUAD, data)
    # im_char.show()
    return im_char


def create_noise_dots_line(image, color, width=3, number=30):
    """
    添加短线 作为噪点
    :param image:原图
    :param color:噪点颜色
    :param width:噪点宽度(大小)
    :param number:(噪点数量)
    :return:修改之后的图片
    """
    draw = ImageDraw.Draw(image)
    w, h = image.size
    while number:
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
        number -= 1
    return image


def create_noise_dots_ellipse(image, color, width=2, number=30):
    """
    按照椭圆的方式添加噪点
    :param image: 原图
    :param color: 噪点的颜色
    :param width: 噪点的宽度
    :param number: 噪点的数量
    :return: 修改之后的图片
    """
    draw = ImageDraw.Draw(image)
    w, h = image.size
    while number:
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        draw.ellipse(((x1, y1), (x1 + width, y1 + width)), fill=color)
        number -= 1
    return image


def create_noise_curve(image, color, width=3):
    """
    添加干扰线(无宽度参数,所以采用上下减一个像素再画一次)
    :param image: 需要处理的图片
    :param width: 干扰线的宽度
    :param color: 干扰线的颜色
    :return: 修改之后的图片
    """
    w, h = image.size
    x1 = random.randint(0, int(w / 5))
    x2 = random.randint(w - int(w / 5), w)
    y1 = random.randint(int(h / 5), 2 * int(h / 5))
    y2 = random.randint(3 * int(h / 5), 4 * int(h / 5))
    points = [x1, y1, x2, y2]
    # 算角度
    end = random.randint(180, 200)
    start = random.randint(0, 20)
    draw = ImageDraw.Draw(image)
    for i in range(width):
        draw.arc([x1, y1 + i, x2, y2 + i], start, end, fill=color)
    return image


def random_color(start, end, opacity=None):
    """
    在区间之间产生颜色
    :param start: 开始像素区间
    :param end: 终止像素区间
    :param opacity: 透明度
    :return: (r,g,b)
    """
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return red, green, blue
    return red, green, blue, opacity


def sin_warp_x(image, amplitude, period, phase, background):
    """
    对图片进行水平方向的波浪处理
    :param image: 需要处理的图片
    :param amplitude: 振幅
    :param period: 周期
    :param phase: 相位
    :param background: 填充的颜色
    :return: 处理之后的图片
    """
    image_w, image_h = image.size
    bg_image = Image.new('RGBA', (image_w, image_h + 2 * amplitude),
                         background)
    unit_length = 6.28318530717958 / period
    offsets = [
        int(amplitude * math.sin(phase * unit_length + unit_length * i))
        for i in range(period)
    ]
    for i in range(image_w - 1):
        box = (i, 0, i + 1, image_h)
        region = image.crop(box)
        bg_image.paste(region, (i, amplitude + offsets[i % period]))
    return bg_image.resize((image_w, image_h), Image.ANTIALIAS)


def sin_warp_y(image, amplitude, period, phase, background):
    """
    对图片进行垂直方向的波浪处理
    :param image: 需要处理的图片
    :param amplitude: 振幅
    :param period: 周期
    :param phase: 相位
    :param background: 填充的颜色
    :return: 处理之后的图片
    """
    image_w, image_h = image.size
    bg_image = Image.new('RGBA', (image_w + 2 * amplitude, image_h),
                         background)
    unit_length = 6.28318530717958 / period
    offsets = [
        int(amplitude *
            math.sin((phase / period) * unit_length + unit_length * i))
        for i in range(period)
    ]
    for i in range(image_h - 1):
        box = (0, i, image_w, i + 1)
        region = image.crop(box)
        bg_image.paste(region, ((amplitude + offsets[i % period]), i))
    return bg_image.resize((image_w, image_h), Image.ANTIALIAS)

def convert_binarization(image, threshold):
    """
    :param image: 需要进行二值化的图片
    :return: 二值化之后的图片
    """
    (image_w, image_h) = image.size
    pixdata = image.load()
    for iter_y in range(image_h):
        for iter_x in range(image_w):
            if pixdata[iter_x, iter_y] < threshold:
                pixdata[iter_x, iter_y] = 0
            else:
                pixdata[iter_x, iter_y] = 255
    return image

