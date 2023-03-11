import base64
import itertools
import json
from io import BytesIO

from PIL import Image


def decode_base64_to_image(encoding: str) -> Image:
    content = encoding.split(";")[1]
    image_encoded = content.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(image_encoded)))


def load_label_mapping(mapping_file_path):
    """
    Load a JSON mapping { class ID -> friendly class name }.
    Used in BaseHandler.
    """

    with open(mapping_file_path) as f:
        mapping = json.load(f)

    if not isinstance(mapping, dict):
        raise Exception(
            'index->name JSON mapping should be in "class": "label" format.'
        )

    # Older examples had a different syntax than others. This code accommodates those.
    if "object_type_names" in mapping and isinstance(
        mapping["object_type_names"], list
    ):
        mapping = {str(k): v for k, v in enumerate(mapping["object_type_names"])}
        return mapping

    for key, value in mapping.items():
        new_value = value
        if isinstance(new_value, list):
            new_value = value[-1]
        if not isinstance(new_value, str):
            raise Exception(
                "labels in index->name mapping must be either str or List[str]"
            )
        mapping[key] = new_value
    return mapping


def map_class_to_label(probs, mapping=None, lbl_classes=None):
    """
    Given a list of classes & probabilities, return a dictionary of
    { friendly class name -> probability }
    """
    if not isinstance(probs, list):
        raise Exception("Convert classes to list before doing mapping")

    if mapping is not None and not isinstance(mapping, dict):
        raise Exception("Mapping must be a dict")

    if lbl_classes is None:
        lbl_classes = itertools.repeat(range(len(probs[0])), len(probs))

    results = [
        {
            (mapping[str(lbl_class)] if mapping is not None else str(lbl_class)): prob
            for lbl_class, prob in zip(*row)
        }
        for row in zip(lbl_classes, probs)
    ]

    return results
