from flask import Flask, request, jsonify
import boto3
from flask_cors import CORS
import base64
import numpy as np
from PIL import Image
import io
import json 
app = Flask(__name__)
CORS(app, origins="*")
# Initialize SageMaker client
sagemaker = boto3.client('sagemaker-runtime', 'us-east-1')
# Define SageMaker endpoint name
endpoint_name = 'sagemaker-inference-pytorch-ml-m5-ml-m5-2024-02-15-07-12-31-153'

def base64_to_bytearray(base64_string):
    # Decode the Base64 string to bytes
    image_bytes = base64.b64decode(base64_string)
    # Convert the bytes to a bytearray
    byte_array = bytearray(image_bytes)
    return byte_array

def base64_to_pil_image(base64_string):
    # Decode the Base64 string to bytes
    image_bytes = base64.b64decode(base64_string)
    # Convert the bytes to a PIL image
    image = Image.open(io.BytesIO(image_bytes))
    new_image = Image.new("RGB", image.size, "white")
    # Paste the original image onto the new image with white background
    new_image.paste(image, (0, 0), image)
    # new_image.show()
    return new_image

def pil_to_bytearray(pil_image):
    # Convert the PIL image to a bytearray
    image_byte_array = io.BytesIO()
    pil_image.save(image_byte_array, format="PNG")
    return image_byte_array.getvalue()


@app.route('/classify', methods=['POST'])
def classify_character():

    try:
        image_data = request.json['image_data']
        white_bg_image = base64_to_pil_image(image_data)
        image_data = pil_to_bytearray(white_bg_image)

        print('calling sagemaker endpoint')
        # # Call SageMaker endpoint for classification
        try:
            response = sagemaker.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/octet-stream',
                Body=image_data
            )
        except Exception as e:
            print('error', e)
            return str(e), 500
        print('called sagemaker endpoint')
        res = response['Body'].read().decode('utf-8')
        res = list(map(np.float32, res[1:-1].split(',')))
        # print("Most likely class: {}".format(np.argmax(res)))
        
        object_categories = {}
        with open("th_chars_classes_48.txt", "r") as f:
            for line in f:
                key, val = line.strip().split(":")
                object_categories[key] = val.strip(" ").strip(",")
        print(
            "The label is",
            object_categories[str(np.argmax(res))],
            "with probability",
            str(np.amax(res))[:5],
        )
        char = object_categories[str(np.argmax(res))].encode('utf-8')
        return json.dumps({'result': char.decode('utf-8'), 'prob': str(np.amax(res))[:5]}), 200
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
