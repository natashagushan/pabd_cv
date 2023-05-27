import os
import shutil

import PIL
import click
import tensorflow
import tf


@click.command()
@click.option('-i', '--in_dir', default='data/raw/kaggle')
@click.option('-o', '--out_dir', default='data/processed/PetImages')
@click.option('-n', '--n_img', default=20)
@click.option('-s', '--img_size', default=180)
def process_data(in_dir, out_dir, n_img, img_size):
    make_out_dir(out_dir)
    copy_imgs(in_dir, out_dir, n_img, img_size)

def make_out_dir(out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, 'Cat'))
    os.mkdir(os.path.join(out_dir, 'Dog'))

def copy_imgs(in_dir, out_dir, n_img, img_size):
    all_imgs = os.listdir(in_dir)
    cat_imgs = [img for img in all_imgs if img.startswith('cat')]
    dog_imgs = [img for img in all_imgs if img.startswith('dog')]

    def resize_and_save(img_list, category_name):
        for cat_img in img_list[:n_img]:
            in_img_path = os.path.join(in_dir, cat_img)
            img = PIL.Image.open(in_img_path)
            img_r = img.resize((img_size, img_size))
            out_img_path = os.path.join(out_dir, category_name, cat_img)
            img_r.save(out_img_path)

    resize_and_save(cat_imgs, "Cat")
    resize_and_save(dog_imgs, "Dog")

def filter_corrupted(out_dir):
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(out_dir, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

if __name__ == '__main__':
    process_data()

