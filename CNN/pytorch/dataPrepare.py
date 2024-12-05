import os
import shutil
import random
# set random seed
random.seed(0)
def split_data(source_dir, train_dir, test_dir, split_ratio=0.7):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    files = os.listdir(source_dir)
    random.shuffle(files)
    
    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    test_files = files[split_index:]

    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))
    
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

if __name__ == "__main__":
    source_directory = '/home/allen/Desktop/Cuda_Deeplearning/data/dogs-vs-cats/original'
    train_directory = '/home/allen/Desktop/Cuda_Deeplearning/data/dogs-vs-cats/train'
    test_directory = '/home/allen/Desktop/Cuda_Deeplearning/data/dogs-vs-cats/test'
    
    split_data(source_directory, train_directory, test_directory)