#!/usr/bin/env python
# coding: utf-8

# # PascalVOC2012から特定のインデックス画像をピックアップする

# In[5]:


INPUT_PATH = '0_input\VOCdevkit\VOC2012'
INPUT_JPEGIMAGE_PATH = '\JPEGImages\\'
INPUT_SEGMENT_PATH = '\SegmentationClass\*'
OUTPUT_SEG_PATH = '1_outputSegmentation\\'
OUTPUT_BASE_PATH = '2_base\\'
OUTPUT_ORI_PATH = OUTPUT_BASE_PATH + 'image\\'
OUTPUT_BINARY_PATH = OUTPUT_BASE_PATH + 'mask\\'
OUTPUT_AUGMENTATION_PATH = '3_augmentation\\'
INDEX_CAT = 8 
INDEX_DOG = 12


# ## 特定のインデックスを抽出&別フォルダに出力

# In[2]:


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
    #targetImageList.extend(getPickUpImage(INDEX_DOG))
    print(len(targetImageList))


# ## 画像を調整  
# 画像サイズ・ アルファチャンネル削除   
# 参考URL  
# [セマンティックセグメンテーションをやってみた](https://qiita.com/yakisobamilk/items/2470354c8d01aaf1e510)
# 

# In[3]:


from PIL import Image
from glob import glob

def reshapeImage(path):
    cnt = 0
    files = glob(path)
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
    cnt = reshapeImage(OUTPUT_SEG_PATH + '*')
    print("reshape [%s] Size : %i" %(OUTPUT_SEG_PATH, cnt))
    cnt = reshapeImage(OUTPUT_ORI_PATH + '*')
    print("reshape [%s] Size : %i" %(OUTPUT_ORI_PATH, cnt))
    


# ## 2値化してフォルダにコピーする  
# インデックスの扱いは以下のURLを参考にした  
# [Numpyでインデックスカラー画像（VOC2012のマスク）→RGB画像への変換をする方法](https://blog.shikoan.com/numpy-indexedcolor-to-rgb/)  
# 新しいパレットを設定するのは以下のURL  
# [インデックスカラーのカラーパレットの編集](https://teratail.com/questions/187368)

# In[4]:


from PIL import Image
import numpy as np
from glob import glob

def binarizationImage(idx):
    ''' 
    指定したインデックス画像のみ残して2値化する
    [input] インデックス
    [output] 変換したファイル数
    '''
    files = glob(OUTPUT_SEG_PATH + '*')
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
            # ターゲットのインデックス以外は[0]に、
            # ターゲットのインデックスは[1]に置き換える
            p_array = np.asarray(im)
            reduced = p_array.copy()
            reduced[reduced != idx] = 0
            reduced[reduced == idx] = 1
            # 画像モードをPに変更する
            #新しいパレットを設定する
            pil_img = Image.fromarray(reduced)
            pil_img.putpalette(palette)
            
            # 別フォルダにコピーする
            pos = str(path).rfind("\\")
            fn =path[pos + 1:]
            pil_img.save(OUTPUT_BINARY_PATH + fn, quality=95)
            cnt += 1
    return cnt
if __name__ == "__main__":
    cnt = binarizationImage(INDEX_CAT)
    print("Binarization Image Size : %i" % cnt)
    


# ## ImageDataGenerator  
#   
# [画像の前処理](https://keras.io/ja/preprocessing/image/)  
# [セマンティックセグメンテーションをやってみた](https://qiita.com/yakisobamilk/items/2470354c8d01aaf1e510)  
# 
# 

# In[13]:


from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def adjustData(img, mask):
    # 元画像の方は255で割って正規化する
    if np.max(img) > 1:
        img = img / 255.

    # マスク画像の方はOne-Hotベクトル化する
    # パレットカラーをndarrayで取得する
    # 0:黒(0,0,0), 1:白(255,255,255)
    palette = np.zeros((2, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    palette[1] = [255, 255, 255]

    # パレットとRGB値を比較してマスク画像をOne-hot化する
    onehot = np.zeros((mask.shape[0], 256, 256, len(palette)), dtype=np.uint8)
    for i in range(len(palette)):
        # 現在カテゴリのRGB値を[R, G, B]の形で取得する
        cat_color = palette[i]

        # 画像が現在カテゴリ色と一致する画素に1を立てた(256, 256)のndarrayを作る
        temp = np.where((mask[:, :, :, 0] == cat_color[0]) &
                        (mask[:, :, :, 1] == cat_color[1]) &
                        (mask[:, :, :, 2] == cat_color[2]), 1, 0)

        # 現在カテゴリに結果を割り当てる
        onehot[:, :, :, i] = temp

    return img, onehot

def trainGenerator(image_folder, batch_size=20, save_to_dir=[None, None]):
    # 2つのジェネレータには同じパラメータを設定する必要がある
    data_gen_args = dict(
        rotation_range=90.,
        width_shift_range=1.,   # 元画像上でのシフト量128にzoom_ratioをかけてint型で設定する
        height_shift_range=1.,  # 同上
        horizontal_flip=True,
        rescale=None            # リスケールはadjustData()でやる
    )
    seed = 1                    # Shuffle時のSeedも共通にしないといけない

    # ImageDataGeneratorを準備
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # ジェネレータを準備
    image_generator = image_datagen.flow_from_directory(
        directory=image_folder,
        classes=['image'],      # directoryの下のフォルダを1つ選び、
        class_mode=None,        # そのクラスだけを読み込んで、正解ラベルは返さない
        target_size=(256, 256),
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir[0]
    )
    mask_generator = mask_datagen.flow_from_directory(
        directory=image_folder,
        classes=['mask'],
        class_mode=None,
        target_size=(256, 256),
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir[1]
    )

    for (img, mask) in zip(image_generator, mask_generator):
        img, mask = adjustData(img, mask)
        yield img, mask    

if __name__ == '__main__':
    temp_gen = trainGenerator(
        OUTPUT_BASE_PATH, 
        batch_size=1, 
        save_to_dir=[ 
            OUTPUT_AUGMENTATION_PATH + 'image', 
            OUTPUT_AUGMENTATION_PATH+ 'mask'
        ]
    )

    DATA_SIZE = 250
    cnt = 0
    for img, mask in temp_gen:
        cnt += 1
        if cnt >= DATA_SIZE:
            break
    
    print("[End]")

