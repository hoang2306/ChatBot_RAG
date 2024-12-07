import os

from env import MUSIC_DATA_FOLDER, DISEASE_DATA_FOLDER
# Đảm bảo all_dir là một chuỗi đường dẫn đến thư mục
all_dir = MUSIC_DATA_FOLDER

for dirpath, dirnames, filenames in os.walk(all_dir):
    for filename in filenames:
        if filename.endswith('.txt'):
            print(f"Found .txt file: {os.path.join(dirpath, filename)}")
