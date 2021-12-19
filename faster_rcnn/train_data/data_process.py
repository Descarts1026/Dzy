import glob
import os
import shutil
target = r"E:\train_data3\bird\VOCdevkit\VOC2007\JPEGImages"
for index,(parent,fold,imgs) in enumerate(os.walk(r"E:\train_data3\bird\VOCdevkit\VOC2007\tmp")):
    if index == 0:
        continue
    for img in imgs:
        if img.endswith(".jpg"):
            shutil.copy(os.path.join(parent,img),os.path.join(target,img))
        else:
            print(os.path.join(parent,img))