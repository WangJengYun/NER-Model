import os 
import shutil
from opencc import OpenCC
cc = OpenCC('s2tw')

import_data_folder = './data/simple_ch/msra/'
saving_data_folder = './data/traditional_ch/msra/'


sub_folders = os.listdir(import_data_folder)

for sub_folder in sub_folders:
    # sub_folder = sub_folders[0]
    read_path = import_data_folder + sub_folder + '/' + 'sentences.txt'
    with open(read_path,'r',encoding="utf-8") as file :
        content = file.read()
    # convert simple ch to tranditional ch 
    saving_content = cc.convert(content)
    
    saving_folder_path = saving_data_folder + sub_folder + '/' 
    if not os.path.exists(saving_folder_path) :
        os.makedirs(saving_folder_path)
    
    saving_file_path = saving_folder_path + 'sentencess.txt'
    with open(saving_file_path,'w',encoding="utf-8") as file :
        file.write(saving_content)

for sub_folder in sub_folders:
    # sub_folder = sub_folders[0]
    read_path = import_data_folder + sub_folder + '/' + 'target.txt'
    saving_file_path =  saving_data_folder + sub_folder + '/'  + 'target.txt'
    shutil.copyfile(read_path,saving_file_path)
    

    
