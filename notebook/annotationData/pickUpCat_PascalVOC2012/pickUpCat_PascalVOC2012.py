#!/usr/bin/env python
# coding: utf-8

# # PascalVOC2012から特定のインデックス画像をピックアップする

# In[1]:


INPUT_PATH = '0_input\VOCdevkit\VOC2012\\'
INPUT_IMAGE_PATH = 'JPEGImages\\'
INPUT_MASK_PATH = 'SegmentationClass\\'
SELECT_PATH = '1_select\\'
BASE_PATH = '2_base\\'
IMAGE_PATH = 'image\\'
MASK_PATH = 'mask\\'
INDEX_CAT = 8 


# ## 特定のインデックスを抽出&別フォルダに出力

# In[3]:


from PIL import Image
import numpy as np
from glob import glob

def getPickUpImage(idx, path):
    ''' 
    指定したインデックス画像名を取得する
    [input] インデックス, パス(inputImage, inputMask, outputImage, outputMask)
    [output] 画像名のList
    '''
    inputImagePath = path[0]
    imputMaskPath = path[1] + '*'
    outputImagePath = path[2]
    outputMaskPath = path[3]
    targetImageList = list()
    files = glob(imputMaskPath)

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
                im.save(outputMaskPath + fn + '.png', quality=95)
                
                # 別フォルダにコピーする(オリジナル画像)
                fileOri = inputImagePath + fn + '.jpg'
                with Image.open(fileOri) as imOri:
                    imOri.save(outputImagePath + fn + '.jpg', quality=95)                

    print("[End] GetSize : %i" % cnt)
    return targetImageList
    
if __name__ == "__main__":
    path = [
        INPUT_PATH + INPUT_IMAGE_PATH,
        INPUT_PATH + INPUT_MASK_PATH,
        SELECT_PATH + IMAGE_PATH,
        SELECT_PATH + MASK_PATH
    ]
    targetImageList = getPickUpImage(INDEX_CAT, path)
    print(len(targetImageList))


# ## 画像を調整  
# 画像サイズ・ アルファチャンネル削除   
# 参考URL  
# [セマンティックセグメンテーションをやってみた](https://qiita.com/yakisobamilk/items/2470354c8d01aaf1e510)
# 

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

            # 512*512にリサイズ
            im = im.resize((512, 512))

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
    cnt = reshapeImage(SELECT_PATH + IMAGE_PATH)
    print("reshape [%s] Size : %i" %(IMAGE_PATH, cnt))
    cnt = reshapeImage(SELECT_PATH + MASK_PATH)
    print("reshape [%s] Size : %i" %(MASK_PATH, cnt))
    


# ## ImageDataGenerator  
#   
# [画像の前処理](https://keras.io/ja/preprocessing/image/)  
# [セマンティックセグメンテーションをやってみた](https://qiita.com/yakisobamilk/items/2470354c8d01aaf1e510)  
# 
# 

# In[9]:


from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def trainGenerator(image_folder, batch_size=20, save_to_dir=[None, None]):
    # 2つのジェネレータには同じパラメータを設定する必要がある
    data_gen_args = dict(
        rotation_range=90.,
        width_shift_range=1.,
        height_shift_range=1.,
        horizontal_flip=True,
        rescale=None
    )
    # Shuffle時のSeedも共通にしないといけない
    seed = 1                    

    # ImageDataGeneratorを準備
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # ジェネレータを準備
    image_generator = image_datagen.flow_from_directory(
        directory = image_folder,
        classes = ['image'], 
        class_mode = None,
        target_size = (256, 256),
        batch_size = batch_size,
        seed = seed,
        save_to_dir = save_to_dir[0],
        save_format = 'jpg'
    )
    mask_generator = mask_datagen.flow_from_directory(
        directory=image_folder,
        classes=['mask'],
        class_mode=None,
        target_size=(256, 256),
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir[1],
        save_format = 'png'
    )
    
    for (img, mask) in zip(image_generator, mask_generator):
        yield img, mask    

if __name__ == '__main__':
    temp_gen = trainGenerator(
        SELECT_PATH, 
        batch_size=1, 
        save_to_dir=[ 
            BASE_PATH + IMAGE_PATH, 
            BASE_PATH + MASK_PATH
        ]
    )

    DATA_SIZE = 250
    cnt = 0
    for img, mask in temp_gen:
        cnt += 1
        if cnt >= DATA_SIZE:
            break
    
    print("[End]")


# ## 2値化してフォルダにコピーする  
# RGBからの2値化は以下のURLを参考にした  
# [python，OpenCV，numpyによる色抽出・変換](https://teratail.com/questions/100301)  
# 
# インデックスの扱いは以下のURLを参考にした  
# [Numpyでインデックスカラー画像（VOC2012のマスク）→RGB画像への変換をする方法](https://blog.shikoan.com/numpy-indexedcolor-to-rgb/)  
# 新しいパレットを設定するのは以下のURL  
# [インデックスカラーのカラーパレットの編集](https://teratail.com/questions/187368)

# In[10]:


from PIL import Image
import numpy as np
from glob import glob

def binarizationImage(path):
    ''' 
    指定したインデックスカラーのみ残して2値化する
    [input] パス(入力,出力)
    [output] 変換したファイル数
    '''
    inputPath = path[0] + '*'
    outputPath = path[1]
    files = glob(inputPath)
    # 新しいパレット 0:黒(0,0,0), 1:白(255,255,255)
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    palette[1] = [255, 255, 255]
    palette = palette.reshape(-1).tolist()

    cnt = 0
    for path in files:
        if path.find('.png') < 0:
            continue
        with Image.open(path) as im:
            if im.mode != 'RGB':
                print("not RGB mode...")
                continue
            
            #print(im.mode)
            im_list = np.asarray(im)
            p = np.asarray(im, dtype=np.uint8)
            reduced = p.copy()
            cond_p = (reduced[:,:,0] == 72) & (reduced[:,:,1] == 0) & (reduced[:,:,2] == 0)
            cond_f = np.logical_not(cond_p)
            reduced[cond_p] = 1
            reduced[cond_f] = 0

            # パレットモードの画像を出力する
            base = np.zeros((256, 256), dtype=np.uint8)
            for i in range(256):
                for j in range(256):
                    if reduced[i,j,0] > 0:
                        base[i,j] = 1
            
            # 別フォルダにコピーする
            with Image.fromarray(base, mode="P") as im:
                # パレットの設定  
                im.putpalette(palette)              
                pos = str(path).rfind("\\")
                fn =path[pos + 1:]
                im.save(outputPath + fn, quality=95)
                cnt += 1
    return cnt
if __name__ == "__main__":
    path = [
        BASE_PATH + MASK_PATH,
        BASE_PATH + MASK_PATH
    ]
    cnt = binarizationImage(path)
    print("Binarization Image Size : %i" % cnt)
    


# ## 2値化のテスト

# In[3]:


import cv2
from PIL import Image
import numpy as np
from glob import glob

# 新しいパレット 0:黒(0,0,0), 1:白(255,255,255)
palette = np.zeros((256, 3), dtype=np.uint8)
palette[0] = [0, 0, 0]
palette[1] = [255, 255, 255]
palette = palette.reshape(-1).tolist()
    
inputPath = BASE_PATH + MASK_PATH + "*"
print(inputPath)
files = glob(inputPath)
for path in files:
    with Image.open(path) as im:
        #print(im.mode)
        im_list = np.asarray(im)
        p = np.asarray(im, dtype=np.uint8)
        reduced = p.copy()
        cond_p = (reduced[:,:,0] == 72) & (reduced[:,:,1] == 0) & (reduced[:,:,2] == 0)
        cond_f = np.logical_not(cond_p)
        reduced[cond_p] = 1
        reduced[cond_f] = 0
        
        # パレットモードの画像を出力する
        base = np.zeros((256, 256), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                if reduced[i,j,0] > 0:
                    base[i,j] = 1

        with Image.fromarray(base, mode="P") as im:
            im.putpalette(palette)  # パレットの設定  
            im.save(path, quality=95)
    

