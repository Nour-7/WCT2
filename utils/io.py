"""
from photo_wct.py of https://github.com/NVIDIA/FastPhotoStyle
Copyright (C) 2018 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0
"""
import os
import datetime

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


class Timer:
    def __init__(self, msg='Elapsed time: {}', verbose=True):
        self.msg = msg
        self.start_time = None
        self.verbose = verbose

    def __enter__(self):
        self.start_time = datetime.datetime.now()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.verbose:
            print(self.msg.format(datetime.datetime.now() - self.start_time))


def open_image(image_path, image_size=None):
    image = Image.open(image_path)
    _transforms = []
    if image_size is not None:
        image = transforms.Resize(image_size)(image)
        # _transforms.append(transforms.Resize(image_size))
    w, h = image.size
    _transforms.append(transforms.CenterCrop((h // 16 * 16, w // 16 * 16)))
    _transforms.append(transforms.ToTensor())
    transform = transforms.Compose(_transforms)
    return transform(image).unsqueeze(0)



def change_seg(seg):


    # color_dict = {
    #     (0, 0, 255): 3,  # blue
    #     (0, 255, 0): 2,  # green
    #     (0, 0, 0): 0,  # black
    #     (255, 255, 255): 1,  # white
    #     (255, 0, 0): 4,  # red
    #     (255, 255, 0): 5,  # yellow
    #     (128, 128, 128): 6,  # grey
    #     (0, 255, 255): 7,  # lightblue
    #     (255, 0, 255): 8  # purple
    # }

    color_dict = {
        (188.955    ,113.985   ,   0.      ) : 0,
        ( 24.99     , 82.875   , 216.75    ) : 1,
        ( 31.875    ,176.97    , 236.895   ) : 2,
        (141.78     , 46.920002, 125.96999 ) : 3,
        ( 47.94     ,171.87001 , 118.829994) : 4,
        (237.91501  ,189.975   ,  76.755   ) : 5,
        ( 46.920002 , 19.890001, 161.925   ) : 6,
        ( 76.5      , 76.5     ,  76.5     ) : 7,
        (153.       ,153.      , 153.      ) : 8,
        (  0.       ,  0.      , 255.      ) : 9,
        (  0.       ,127.5     , 255.      ) : 10,
        (  0.       ,190.99501 , 190.99501 ) : 11,
        (  0.       ,255.      ,   0.      ) : 12,
        (255.       ,  0.      ,   0.      ) : 13,
        (255.       ,  0.      , 170.08499 ) : 14,
        (  0.       , 84.915   ,  84.915   ) : 15,
        (  0.       ,170.08499 ,  84.915   ) : 16,
        (  0.       ,255.      ,  84.915   ) : 17,
        (  0.       , 84.915   , 170.08499 ) : 18,
        (  0.       ,170.08499 , 170.08499 ) : 19,
        (  0.       ,255.      , 170.08499 ) : 20,
        (  0.       , 84.915   , 255.      ) : 21,
        (  0.       ,170.08499 , 255.      ) : 22,
        (  0.       ,255.      , 255.      ) : 23,
        (127.5      , 84.915   ,   0.      ) : 24,
        (127.5      ,170.08499 ,   0.      ) : 25,
        (127.5      ,255.      ,   0.      ) : 26,
        (127.5      ,  0.      ,  84.915   ) : 27,
        (127.5      , 84.915   ,  84.915   ) : 28,
        (127.5      ,170.08499 ,  84.915   ) : 29,
        (127.5      ,255.      ,  84.915   ) : 30,
        (127.5      ,  0.      , 170.08499 ) : 31,
        (127.5      , 84.915   , 170.08499 ) : 32,
        (127.5      ,170.08499 , 170.08499 ) : 33,
        (127.5      ,255.      , 170.08499 ) : 34,
        (127.5      ,  0.      , 255.      ) : 35,
        (127.5      , 84.915   , 255.      ) : 36,
        (127.5      ,170.08499 , 255.      ) : 37,
        (127.5      ,255.      , 255.      ) : 38,
        (255.       , 84.915   ,   0.      ) : 39,
        (255.       ,170.08499 ,   0.      ) : 40,
        (255.       ,255.      ,   0.      ) : 41,
        (255.       ,  0.      ,  84.915   ) : 42,
        (255.       , 84.915   ,  84.915   ) : 43,
        (255.       ,170.08499 ,  84.915   ) : 44,
        (255.       ,255.      ,  84.915   ) : 45,
        (255.       ,  0.      , 170.08499 ) : 46,
        (255.       , 84.915   , 170.08499 ) : 47,
        (255.       ,170.08499 , 170.08499 ) : 48,
        (255.       ,255.      , 170.08499 ) : 49,
        (255.       ,  0.      , 255.      ) : 50,
        (255.       , 84.915   , 255.      ) : 51,
        (255.       ,170.08499 , 255.      ) : 52,
        (  0.       ,  0.      ,  84.915   ) : 53,
        (  0.       ,  0.      , 127.5     ) : 54,
        (  0.       ,  0.      , 170.08499 ) : 55,
        (  0.       ,  0.      , 212.41501 ) : 56,
        (  0.       ,  0.      , 255.      ) : 57,
        (  0.       , 42.585   ,   0.      ) : 58,
        (  0.       , 84.915   ,   0.      ) : 59,
        (  0.       ,127.5     ,   0.      ) : 60,
        (  0.       ,170.08499 ,   0.      ) : 61,
        (  0.       ,212.41501 ,   0.      ) : 62,
        (  0.       ,255.      ,   0.      ) : 63,
        ( 42.585    ,  0.      ,   0.      ) : 64,
        ( 84.915    ,  0.      ,   0.      ) : 65,
        (127.5      ,  0.      ,   0.      ) : 66,
        (170.08499  ,  0.      ,   0.      ) : 67,
        (212.41501  ,  0.      ,   0.      ) : 68,
        (255.       ,  0.      ,   0.      ) : 69,
        (  0.       ,  0.      ,   0.      ) : 70,
        ( 36.465    , 36.465   ,  36.465   ) : 71,
        (218.535    ,218.535   , 218.535   ) : 72,
        (255.       ,255.      , 255.      ) : 73
    }

    arr_seg = np.asarray(seg)
    new_seg = np.zeros(arr_seg.shape[:-1])
    for x in range(arr_seg.shape[0]):
        for y in range(arr_seg.shape[1]):
            if tuple(arr_seg[x, y, :]) in color_dict:
                new_seg[x, y] = color_dict[tuple(arr_seg[x, y, :])]
            else:
                min_dist_index = 0
                min_dist = 99999
                for key in color_dict:
                    dist = np.sum(np.abs(np.asarray(key) - arr_seg[x, y, :]))
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_index = color_dict[key]
                    elif dist == min_dist:
                        try:
                            min_dist_index = new_seg[x, y-1, :]
                        except Exception:
                            pass
                new_seg[x, y] = min_dist_index
    return new_seg.astype(np.uint8)


def load_segment(image_path, image_size=None):
    l = 1
    if not image_path:
        return np.asarray([])
    image = Image.open(image_path)
    if image_size is not None:
        transform = transforms.Resize(image_size, interpolation=Image.NEAREST)
        image = transform(image)
    w, h = image.size
    transform = transforms.CenterCrop((h // 16 * 16, w // 16 * 16))
    image = transform(image)
    if len(np.asarray(image).shape) == 3:
        image = change_seg(image)

    Image.SAVE(image, 'img'+str(l)+'.png')
    l += 1
    return np.asarray(image)


def compute_label_info(content_segment, style_segment):
    if not content_segment.size or not style_segment.size:
        return None, None
    max_label = np.max(content_segment) + 1
    label_set = np.unique(content_segment)
    label_indicator = np.zeros(max_label)
    for l in label_set:
        content_mask = np.where(content_segment.reshape(content_segment.shape[0] * content_segment.shape[1]) == l)
        style_mask = np.where(style_segment.reshape(style_segment.shape[0] * style_segment.shape[1]) == l)

        c_size = content_mask[0].size
        s_size = style_mask[0].size
        if c_size > 10 and s_size > 10 and c_size / s_size < 100 and s_size / c_size < 100:
            label_indicator[l] = True
        else:
            label_indicator[l] = False
    return label_set, label_indicator


def mkdir(dname):
    if not os.path.exists(dname):
        os.makedirs(dname)
    else:
        assert os.path.isdir(dname), 'alread exists filename {}'.format(dname)
