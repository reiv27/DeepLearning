import os
import shutil

# Path to folder with class folders from root (/)
dir_path = '/home/user/Documents/practice/tripletloss/data/my_data_2/train'

for dir in os.listdir(dir_path):
    count = 0
    for file in os.listdir(f'{dir_path}/{dir}'):
        if os.path.isfile(os.path.join(f'{dir_path}/{dir}', file)):
            count += 1
    if count < 30:
        shutil.rmtree(f'{dir_path}/{dir}')

print(len(os.listdir(dir_path)))
