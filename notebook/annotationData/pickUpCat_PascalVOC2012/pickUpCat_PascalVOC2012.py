#!/usr/bin/env python
# coding: utf-8

# # PascalVOC2012から特定のインデックス画像をピックアップする

# In[1]:


INPUT_PATH = '0_input\VOCdevkit\VOC2012'
INPUT_JPEGIMAGE_PATH = '\JPEGImages\*'
INPUT_SEGMENT_PATH = '\SegmentationClass\*'
INDEX_CAT = 8 


# ## 特定のインデックスを抽出する  

# In[3]:


from PIL import Image
import numpy as np

from glob import glob

def getPickUpImageList(idx):
    ''' 
    指定したインデックス画像名を取得する
    [input] インデックス
    [output] 画像名のList
    '''
    targetImageList = list()
    files = glob(INPUT_PATH + INPUT_SEGMENT_PATH);

    # パレット(Numpy配列)を取得
    palette = np.array("")
    for path in files:
        with Image.open(path) as im:
            palette = np.array(im.getpalette(), dtype=np.uint8).reshape(-1, 3)
            print("palette size : %s" % str(palette.shape))
            print("target Idx : %i, RGB%s" %(idx, str(palette[idx])))
            break

    cnt = 0
    for path in files:
        with Image.open(path) as im:
            # RGBに変換
            #converted_rgb = np.asarray(im.convert("RGB"))
            #print(converted_rgb.shape)
            
            # ターゲットのインデックス以外の値は[0]に置き換える
            p_array = np.asarray(im)
            reduced = p_array.copy()
            reduced[reduced != idx] = 0
            # 重複データを削除
            uniqueArray = np.unique(reduced, axis=0)
            # ターゲットのインデックスがない場合 -> 全てゼロ -> shapeは(1,500)
            # ターゲットのインデックスがある場合 -> 2行以上ある -> shape[0] > 1
            if uniqueArray.shape[0] > 1:
                cnt += 1
                pos = str(path).rfind("\\")
                fn =path[pos+1:-4]
                #print(fn)                    
                targetImageList.append(fn)

    print("[End] GetSize : %i" % cnt)
    return targetImageList
    
if __name__ == "__main__":
    targetImageList = getPickUpImageList(INDEX_CAT)
    print(len(targetImageList))

