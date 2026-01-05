# -*- coding: utf-8 -*-
"""
@File    :   visualizer.py
@Time    :   2022/04/05 11:39:33
@Author  :   Shilong Liu 
@Contact :   slongliu86@gmail.com
"""

import datetime
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import jittor as jt
from matplotlib import transforms
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils


def renorm(
    img: jt.Var, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
) -> jt.Var:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.ndim == 3 or img.ndim == 4, "img.ndim should be 3 or 4 but %d" % img.ndim
    if img.ndim == 3:
        # (3, H, W) -> (H, W, 3)
        assert img.shape[0] == 3, 'img.shape[0] should be 3 but "%d". (%s)' % (
            img.shape[0],
            str(img.shape),
        )
        img_perm = img.permute(1, 2, 0)
        mean = jt.float32(mean)
        std = jt.float32(std)
        # broadcast along channel dimension
        img_res: jt.Var = img_perm * std + mean
        return img_res.permute(2, 0, 1)
    else:  # img.dim() == 4
        assert img.shape[1] == 3, 'img.shape[1] should be 3 but "%d". (%s)' % (
            img.shape[1],
            str(img.shape),
        )
        img_perm = img.permute(0, 2, 3, 1)
        mean = jt.float32(mean)
        std = jt.float32(std)
        img_res: jt.Var = img_perm * std + mean
        return img_res.permute(0, 3, 1, 2)


class ColorMap:
    def __init__(self, basergb=[255, 255, 0]):
        self.basergb = np.array(basergb)

    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1)  # h, w, 3
        attn1 = attnmap.copy()[..., None]  # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res


class COCOVisualizer():
    def __init__(self) -> None:
        pass

    def visualize(self, img, tgt, caption=None, dpi=120, savedir=None, show_in_console=True):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = '5'
        ax = plt.gca()
        img = renorm(img).permute(1, 2, 0)
        ax.imshow(img)
        
        self.addtgt(tgt)
        if show_in_console:
            plt.show()

        if savedir is not None:
            if caption is None:
                savename = '{}/{}-{}.png'.format(savedir, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
            else:
                savename = '{}/{}-{}-{}.png'.format(savedir, caption, int(tgt['image_id']), str(datetime.datetime.now()).replace(' ', '-'))
            print("savename: {}".format(savename))
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.savefig(savename)
        plt.close()

    def addtgt(self, tgt):
        """
        - tgt: dict. args:
            - boxes: num_boxes, 4. xywh, [0,1].
            - box_label: num_boxes.
        """
        assert 'boxes' in tgt
        ax = plt.gca()
        H, W = tgt['size'].tolist() 
        numbox = tgt['boxes'].shape[0]

        color = []
        polygons = []
        boxes = []
        for box in tgt['boxes'].cpu():
            unnormbbox = box * jt.array([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons.append(Polygon(np_poly))
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            color.append(c)

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)


        if 'box_label' in tgt:
            assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
            for idx, bl in enumerate(tgt['box_label']):
                _string = str(bl)
                bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
                ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1})

        if 'caption' in tgt:
            ax.set_title(tgt['caption'], wrap=True)
