#!/usr/bin/env python
# coding: utf-8

# # PascalVOC2012から特定のインデックス画像をピックアップする

# In[15]:


INPUT_PATH = '0_input\VOCdevkit\VOC2012'
INPUT_JPEGIMAGE_PATH = '\JPEGImages\\'
INPUT_SEGMENT_PATH = '\SegmentationClass\*'
OUTPUT_SEG_PATH = '1_outputSegmentation\\'
OUTPUT_ORI_PATH = '2_outputOriginal\\'
OUTPUT_BINARY_PATH = '3_binarization\\'
INDEX_CAT = 8 
INDEX_DOG = 12


# ## 特定のインデックスを抽出&別フォルダに出力

# In[14]:


from PIL import Image
import numpy as np
from glob import glob

def getPickUpImage(idx):
    ''' 
    指定したインデックス画像名を取得する
    [input] インデックス
    [output] 画像名のList
    '''
    targetImageList = list()
    files = glob(INPUT_PATH + INPUT_SEGMENT_PATH)

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
                fn =path[pos + 1:-4]
                #print(fn)                    
                targetImageList.append(fn)
                
                # 別フォルダにコピーする
                im.save(OUTPUT_SEG_PATH + fn + '.png', quality=95)
                # 別フォルダにコピーする(オリジナル画像)
                fileOri = INPUT_PATH + INPUT_JPEGIMAGE_PATH + fn + '.jpg'
                with Image.open(fileOri) as imOri:
                    imOri.save(OUTPUT_ORI_PATH + fn + '.jpg', quality=95)                

    print("[End] GetSize : %i" % cnt)
    return targetImageList
    
if __name__ == "__main__":
    targetImageList = getPickUpImage(INDEX_CAT)
    targetImageList.extend(getPickUpImage(INDEX_DOG))
    print(len(targetImageList))


# ## 2値化してフォルダにコピーする  
# インデックスの扱いは以下のURLを参考にした  
# [Numpyでインデックスカラー画像（VOC2012のマスク）→RGB画像への変換をする方法](https://blog.shikoan.com/numpy-indexedcolor-to-rgb/)

# In[36]:


from PIL import Image
import numpy as np
from glob import glob

def binarizationImage(idx):
    ''' 
    指定したインデックス画像のみ残して2値化する
    [input] インデックス
    [output]
    '''
    files = glob(OUTPUT_SEG_PATH + '*')

    # パレット(Numpy配列)を取得
    palette = np.array("")
    for path in files:
        if path.find('.png') < 0:
            continue        
        with Image.open(path) as im:
            palette = np.array(im.getpalette(), dtype=np.uint8).reshape(-1, 3)
            print("palette size : %s" % str(palette.shape))
            print("target Idx : %i, RGB%s" %(idx, str(palette[idx])))
            break

    for path in files:
        if path.find('.png') < 0:
            continue
        with Image.open(path) as im:
            # ターゲットのインデックス以外は[0]に、
            # ターゲットのインデックスは[1]に置き換える
            p_array = np.asarray(im)
            reduced = p_array.copy()
            reduced[reduced != idx] = 0
            reduced[reduced == idx] = 1
            expanded_img = np.eye(palette.shape[0], dtype=np.int32)[reduced]
            use_pallete = palette[:palette.shape[0]].astype(np.int32)
            rgb_array = np.dot(expanded_img, use_pallete).astype(np.uint8)

            # 別フォルダにコピーする
            pos = str(path).rfind("\\")
            fn =path[pos + 1:]
            pil_img = Image.fromarray(rgb_array)
            pil_img.save(OUTPUT_BINARY_PATH + fn, quality=95)
                
if __name__ == "__main__":
    binarizationImage(INDEX_CAT)
    

