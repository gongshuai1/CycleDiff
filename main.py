from optimization.arguments import get_arguments
from optimization.attribute_editor import AttributeEditor
import sys
sys.path.append(".")
sys.path.append("..")


if __name__ == '__main__':
    args = get_arguments()
    attribute_editor = AttributeEditor(args)
    # attribute_editor.edit_image_for_person()
    attribute_editor.edit_image_by_attribute()
    # attribute_editor.sample_batch()
