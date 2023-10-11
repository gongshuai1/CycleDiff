"""
Preprocess for attributes
convert list_attr_celeba.txt to attribute.pt for 30000 images1 in CelebA-HQ dataset
"""
import numpy as np
import os
import torch


def main():
    # Load attribute file
    attribute_file_dir = "/training_data/CelebA/Anno/list_attr_celeba.txt"
    attributes = []
    with open(attribute_file_dir, 'r') as file:
        fhs = file.readlines()
        for index in range(2, len(fhs)):
            line = fhs[index]
            strs = [item.replace(' ', '') for item in line.split(' ')]
            while '' in strs:
                strs.remove('')
            orig_file = strs[0]
            attribute = np.array(strs[1:], dtype=np.float32)
            attributes.append((orig_file, attribute))
        file.close()

    # Load image file
    image_file_dir = "/training_data/CelebA/celeba-hq/celeba-256-tmp"
    new_attributes = []
    index_mapping = dict()
    file_name_list = os.listdir(image_file_dir)
    for i in range(len(file_name_list)):
        file_name = file_name_list[i]

        # Get attribute
        orig_index = int(file_name[0:6])
        orig_file_name, attribute = attributes[orig_index-1]
        attribute = torch.from_numpy(attribute)
        assert orig_file_name == file_name
        new_attributes.append(attribute)

        # index mapping
        new_file_name = '%05d.jpg' % i
        index_mapping[new_file_name] = file_name
        # Rename file
        os.rename(os.path.join(image_file_dir, file_name), os.path.join(image_file_dir, new_file_name))

    # Save attributes
    new_attributes = torch.stack(new_attributes, dim=0)
    torch.save(new_attributes, os.path.join(image_file_dir, 'attribute.pt'))

    # Save index mapping
    with open(os.path.join(image_file_dir, 'index_mapping.txt'), 'a') as file:
        for (key, value) in index_mapping.items():
            file.write(f'{key} = {value}\n')
        file.close()


def attribute_20():
    attribute_path = os.path.join("/training_data/CelebA/celeba-hq-30000/celeba-256-tmp", 'attribute.pt')
    attributes = torch.load(attribute_path)
    attributes = (attributes + 1) // 2
    attributes_20 = [
        torch.from_numpy(
            np.array([attribute[16],  # Goatee - 山羊胡子
                      1. if (attribute[0] + attribute[22] + attribute[30]) > 0 else 0.,  # Beard - 胡子
                      attribute[3],  # Bags_Under_Eyes - 眼袋
                      1. if (attribute[4] + attribute[28]) else 0.,  # Bald - 秃头
                      attribute[5],  # Bangs - 刘海
                      attribute[8],  # Black_Hair - 黑发
                      attribute[9],  # Blond_Hair - 金发
                      attribute[11],  # Brown_Hair - 棕发
                      attribute[17],  # Gray_Hair - 灰发或白发
                      attribute[15],  # Eyeglasses - 眼睛
                      attribute[18],  # Heavy_Makeup - 浓妆
                      attribute[20],  # Male - 男性
                      attribute[21],  # Mouth_Slightly_Open - 微微张开嘴巴
                      attribute[23],  # Narrow_Eyes - 细长的眼睛
                      attribute[26],  # Pale_Skin - 苍白的皮肤
                      attribute[31],  # Smiling - 微笑
                      attribute[32],  # Straight_Hair - 直发
                      attribute[33],  # Wavy_Hair - 卷发
                      attribute[36],  # Wearing_Lipstick - 涂了唇膏
                      attribute[39]], dtype=np.float32))  # Young - 年轻人
        for attribute in attributes
    ]
    attributes_20 = torch.stack(attributes_20, dim=0)
    print(f'attributes_20.shape = {attributes_20.shape}')
    print(f'attributes_20[0] = {attributes_20[0]}')
    torch.save(attributes_20, os.path.join("/training_data/CelebA/celeba-hq-30000/celeba-256-tmp", 'attribute_20_30000.pt'))


def mv_images_256():
    origin_dir = "/training_data/CelebA/celeba-hq/celeba-256-177"
    target_dir = "/training_data/CelebA/celeba-hq/test"

    file_name_list = os.listdir(origin_dir)
    count = 0
    for i in range(len(file_name_list)):
        file_name = file_name_list[i]

        cmd = f"cp {os.path.join(origin_dir, file_name)} {os.path.join(target_dir, file_name)}"
        print(f'cmd = {cmd}')
        os.system(cmd)
        count += 1

    print(f'{count} files have been moved.')


if __name__ == '__main__':
    # main()
    # attribute_20()
    mv_images_256()
