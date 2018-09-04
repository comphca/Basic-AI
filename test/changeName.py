import os
import re
root = "C:/Users/comph/Desktop/dolphins-and-seahorses/seahorse"
listdir = os.listdir(root)
print(listdir)
len = len(listdir)
for i in range(len):
    new_filename_token = []
    new_filename_token.append("1." + str(i+1))
    new_filename_token.append(".jpg")
    new_filename = ''.join(new_filename_token)
    old_filen_path = os.path.join(root,listdir[i])
    new_file_path = os.path.join(root,new_filename)
    os.rename(old_filen_path,new_file_path)

