#!/usr/bin/env python
# coding: utf-8

# # NNCで使うデータセットリストを作成  
# 
# 以下のようなcsvファイルを作成  
# pascal_voc_2012_seg_train_125px.csv
# pascal_voc_2012_seg_val_125px.csv  
# 
# x:image,y:label  
# ./images_125px/2007_000032.png,./labels_125px/2007_000032.png  
# 

# In[1]:


TRAIN_CSV = 'u-net_train.csv'
VAL_CSV = 'u-net_val.csv'

TRAINING_PATH = 'trainingData'
IMAGE_PATH = 'image'
MASK_PATH = 'mask'


# In[4]:


from PIL import Image
from glob import glob

def reshapeImage(path):
    cnt = 0
    files = glob(path + '*')
    for path in files:
        if path.find('.png') < 0 and path.find('.jpg') < 0:
            continue        
        with Image.open(path) as im:                
            # 画像の短辺に合わせて正方形化
            im = crop_to_square(im)

            # 125*125にリサイズ
            im = im.resize((125, 125))

            # アルファチャネルがあればRGBに変換
            if im.mode == 'RGBA':
                im = im.convert('RGB')

            im.save(path, quality=95)
            
            cnt += 1
    return cnt
def crop_to_square(image):
    ''' 画像の短辺に合わせて正方形化
    '''
    size = min(image.size)
    left, upper = (image.width - size) // 2, (image.height - size) // 2
    right, bottom = (image.width + size) // 2, (image.height + size) // 2

    return image.crop((left, upper, right, bottom))

if __name__ == "__main__":
    cnt = reshapeImage(TRAINING_PATH + "\\" + IMAGE_PATH + "\\")
    print("reshape [%s] Size : %i" %(IMAGE_PATH, cnt))
    cnt = reshapeImage(TRAINING_PATH + "\\" + MASK_PATH + "\\")
    print("reshape [%s] Size : %i" %(MASK_PATH, cnt))
    


# In[5]:


import os
import csv
from glob import glob
import random

# ファイル名を取得
fnList = list()
files = glob(TRAINING_PATH + os.sep + IMAGE_PATH + os.sep + "*")
for path in files:
    pos = str(path).rfind("\\")
    fn =path[pos + 1:-4]
    fnList.append(fn)

# シャッフル
random.shuffle(fnList)

# Train : Validation = 7 : 3 に分ける
trainCnt = (int)(len(fnList) / 10 * 7)
valCnt = len(fnList) - trainCnt
print("Data(%i) -> Train(%i), Val(%i)" % (len(fnList), trainCnt, valCnt))

with open('./' + TRAIN_CSV, 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x:image", "y:label"])
    for i in range(trainCnt):
        imagePath = './' + TRAINING_PATH + '/' + IMAGE_PATH + '/' + fnList[i] + '.jpg'
        maskPath = './' + TRAINING_PATH + '/' + MASK_PATH + '/' + fnList[i] + '.png'
        writer.writerow([imagePath, maskPath])

with open('./' + VAL_CSV, 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x:image", "y:label"])
    for i in range(trainCnt, len(fnList)):
        imagePath = './' + TRAINING_PATH + '/' + IMAGE_PATH + '/' + fnList[i] + '.jpg'
        maskPath = './' + TRAINING_PATH + '/' + MASK_PATH + '/' + fnList[i] + '.png'
        writer.writerow([imagePath, maskPath])

