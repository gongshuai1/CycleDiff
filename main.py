from optimization.arguments import get_arguments
from optimization.attribute_editor import AttributeEditor
import sys
import torch
sys.path.append(".")
sys.path.append("..")


if __name__ == '__main__':
    args = get_arguments()
    attribute_editor = AttributeEditor(args)
    # attribute_editor.edit_image_for_person()
    attribute_editor.edit_image_by_attribute()
    # attribute_editor.sample_batch()

    # path = '/training_data/CelebA/celeba-hq-30000/celeba-256-tmp/attribute_20_30000.pt'
    # attributes = torch.load(path)
    # print(f'attributes[8581] = {attributes[8581]}')
    # print(f'attributes[20344] = {attributes[20344]}')
