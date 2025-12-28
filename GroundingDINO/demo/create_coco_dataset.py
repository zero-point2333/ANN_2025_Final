import numpy as np
from pycocotools.coco import COCO
import requests
import zipfile
import os

# 设置数据路径
dataDir = 'coco_dataset'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)

# 初始化COCO api
coco=COCO(annFile)

# 下载图片数据
imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
img_url = img['coco_url']
r = requests.get(img_url, allow_redirects=True)
open('coco_sample_image.jpg', 'wb').write(r.content)

# 解压缩文件
with zipfile.ZipFile('coco_sample_image.zip', 'r') as zip_ref:
    zip_ref.extractall(dataDir)
