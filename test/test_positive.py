import base64
import json
from pathlib import Path

import requests
from requests import Response

url = ""


def test_endpoint():
    image_path = "/images/examples/pred.jpeg"
    with open(image_path, "rb") as f:
        ext = image_path.split(".")[-1]
        prefix = f"data:image/{ext};base64,"
        base64_data = prefix + base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({"body": [base64_data]})

    headers = {"Content-Type": "application/json"}

    response: Response = requests.request(
        "POST", url, headers=headers, data=payload, timeout=15
    )
    print(f"response: {response.text}")
    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert data["statusCode"] == 200
