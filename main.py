from optimization.arguments import get_arguments
from optimization.attribute_editor import AttributeEditor
import sys
import torch
sys.path.append(".")
sys.path.append("..")

if __name__ == '__main__':
    args = get_arguments()
    attribute_editor = AttributeEditor(args)
    # 单人多属性（指定人） - 图五
    # attribute_editor.edit_image_for_person()
    # 单人单属性（指定修改人和属性）
    attribute_editor.edit_image_by_attribute()
    # 所有属性（无需指定） - 八个文件夹
    # attribute_editor.sample_batch()

    # path = '/training_data/CelebA/celeba-hq-30000/celeba-256-tmp/attribute_20_30000.pt'
    # attributes = torch.load(path)
    # print(f'attributes[8581] = {attributes[8581]}')
    # print(f'attributes[20344] = {attributes[20344]}')
