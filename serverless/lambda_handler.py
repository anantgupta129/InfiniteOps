import json
from functools import partial
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from alibi_detect.cd import MMDDriftOnline
from alibi_detect.cd.pytorch import preprocess_drift
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import (decode_base64_to_image, load_label_mapping,
                   map_class_to_label)


topk = 5
categories = load_label_mapping("index_to_name.json")
predict_transforms = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainset_ref = ImageFolder('data/', transform=predict_transforms)
trainset_ref = next(iter(DataLoader(trainset_ref, batch_size=10, shuffle=False)))
print(trainset_ref[0].shape, trainset_ref[1].shape)

model = torch.jit.load("model.scripted.pt").to(device)
model.eval()

preprocess_fn = partial(
    preprocess_drift, model=model, device=device, batch_size=512
)
cd = MMDDriftOnline(
    trainset_ref[0],
    backend="pytorch",
    window_size=1,
    ert=2,
    preprocess_fn=preprocess_fn,
)

response_headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}


def inference(image: Image) -> Dict[str, int]:
    img_tensor = predict_transforms(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        preds = F.softmax(logits, dim=-1)

    print(cd.predict(img_tensor.numpy()))
    return preds


def handle_request(event, context):
    print("----- NEW REQUEST -----")
    print(f"Lambda function ARN: {context.invoked_function_arn}")
    print(f"Lambda funtion version: {context.function_version}")
    print(f"Lambda Request ID: {context.aws_request_id}")

    print(f"Got event", event)

    img_b64 = event["body"]

    try:
        image = decode_base64_to_image(img_b64)

        predictions = inference(image)

        probs, classes = torch.topk(predictions, topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()

        print(f"Lambda time remaining in MS: {context.get_remaining_time_in_millis()}")

        class_to_label = map_class_to_label(probs, categories, classes)
        print(class_to_label)
        
        return {
            "statusCode": 200,
            "headers": response_headers,
            "body": json.dumps(class_to_label),
        }

    except Exception as e:
        print(e)

        return {
            "statusCode": 500,
            "headers": response_headers,
            "body": json.dumps({"message": "Failed to process image: {}".format(e)}),
        }
