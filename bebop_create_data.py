import os
from PIL import Image
import io
import random
import time
import multiprocessing

def resize_pixel(image_file,output_file, width_size, height_size):
    size = width_size, height_size
    for (path,dir,files) in os.walk(image_file):
        for file_name in files:
            try:
                new_img = Image.new("RGB", (width_size, height_size), "black")

                fd = io.open(image_file+file_name,'rb')
                im = Image.open(fd)
                im.thumbnail(size, Image.ANTIALIAS)

                load_img = im.load()
                load_new_img = new_img.load()

                i_offset = (width_size - im.size[0]) / 2
                j_offset = (height_size - im.size[1]) / 2

                for i in range(0, im.size[0]):
                    for j in range(0, im.size[1]):
                        load_new_img[i + i_offset, j + j_offset] = load_img[i, j]

                new_img.save(output_file+file_name,'JPEG')
                fd.close()

            except Exception as e:
                print("[Error]%s: Image writing error: %s" %(image_file+file_name, str(e)))

def change_color(image_file, output_path):
    for (path,dir,files) in os.walk(image_file):
        for file_name in files:
            try:
                fd = io.open(image_file+file_name,'rb')
                im = Image.open(fd).convert('LA')
                im.save(output_path + file_name)

            except Exception as e:
                print("[Error]%s: Image writing error: %s" %(image_file, str(e)))
            fd.close()

def class_label(image_file, output_path, kinds):
    if kinds=="train":
        fd = io.open(output_path + 'bebop2_train_data.txt', 'a')
    elif kinds=="validation":
        fd = io.open(output_path + 'bebop2_validation_data.txt', 'a')

    for (path,dir,files) in os.walk(image_file):
        for file_name in files:
            label_name = image_file.split('/')[-2]
            data = path + file_name + "," + label_name + "\n"
            fd.write(data)
    fd.close()

def class_test_label(image_file, output_path):

    fd = io.open(output_path + 'bebop2_test_data.txt', 'w')

    for (path,dir,files) in os.walk(image_file):
        for file_name in files:
            data = path + file_name + "\n"
            fd.write(data)
    fd.close()

def shuffle_label(label_file, output_path):
    lines = open(label_file,'r').readlines()
    random.shuffle(lines)
    open(output_path, 'w').writelines(lines)

def run(x):
    path_train_bp = "./images/train/before_preprocess/"
    path_train_ap = "./images/train/after_preprocess/"
    path_train_bp_v = "./images/train/before_preprocess/validation_data/"
    path_train_ap_v = "./images/train/after_preprocess/validation_data/"

    # train
    resize_pixel(path_train_bp + x, path_train_bp + "after_resize_pixel/" + x, 96, 96)
    change_color(path_train_bp + "after_resize_pixel/" + x, path_train_ap + x)
    class_label(path_train_ap + x, "./train_data/", "train")

    # validation
    resize_pixel(path_train_bp_v + x, path_train_bp_v + "after_resize_pixel/" + x, 96, 96)
    change_color(path_train_bp_v + "after_resize_pixel/" + x, path_train_ap_v + x)
    class_label(path_train_ap_v + x, "./train_data/", "validation")

if __name__ == "__main__":
    start_time = int(time.time())

    pool = multiprocessing.Pool(8)
    where_to_go = ["forward/", "turn_left/", "turn_right/"]
    pool.map(run,where_to_go)

    print("preprocess done!: %s" % (time.time() - start_time), "seconds")
