from os import path as osp
from PIL import Image
import glob

from basicsr.utils import scandir


def generate_meta_info_Jilin189(mode='train'):
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = '../dataset/Jilin189/{}/GT'.format(mode)
    meta_info_txt = 'basicsr/data/meta_info/meta_info_Jilin189_{}_GT.txt'.format(mode)

    video_list = sorted(glob.glob(osp.join(gt_folder, '*')))

    with open(meta_info_txt, 'w') as f:
        for idx, video_path in enumerate(video_list):
            imgs = glob.glob(osp.join(video_path, '*.png'))
            if len(imgs) == 0:
                imgs = glob.glob(osp.join(video_path, '*.jpg'))
            img = Image.open(imgs[0])  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')
            lenth = len(imgs)
            info = f'{video_path.split("/")[-1]} {lenth} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


def generate_meta_info_svsr(mode='train'):
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = '../dataset/SVSR400/{}/GT'.format(mode)
    meta_info_txt = 'basicsr/data/meta_info/meta_info_SVSR400_{}_GT.txt'.format(mode)

    video_list = sorted(glob.glob(osp.join(gt_folder, '*')))

    with open(meta_info_txt, 'w') as f:
        for idx, video_path in enumerate(video_list):
            imgs = glob.glob(osp.join(video_path, '*.png'))
            if len(imgs) == 0:
                imgs = glob.glob(osp.join(video_path, '*.jpg'))
            img = Image.open(imgs[0])  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')
            lenth = len(imgs)
            info = f'{video_path.split("/")[-1]} {lenth} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = 'datasets/DIV2K/DIV2K_train_HR_sub/'
    meta_info_txt = 'basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


if __name__ == '__main__':
    # generate_meta_info_div2k()

    generate_meta_info_svsr('train')
    generate_meta_info_svsr('val')

    # generate_meta_info_Jilin189('train')
    # generate_meta_info_Jilin189('eval')
