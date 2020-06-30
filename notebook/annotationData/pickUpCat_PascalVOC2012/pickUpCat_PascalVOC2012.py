#!/usr/bin/env python
# coding: utf-8

# # PascalVOC2012からネコ画像だけピックアップする

# In[13]:


INPUT_PATH = '0_input\VOCdevkit\VOC2012'
INPUT_JPEGIMAGE_PATH = '\JPEGImages\*'
INPUT_SEGMENT_PATH = '\SegmentationClass\*'


# In[15]:


from PIL import Image
from glob import glob
files = glob(INPUT_PATH + INPUT_SEGMENT_PATH);
for path in files:
    with open(path) as f:
        print(f)
        #image_sample_palette = Image.open(f)

