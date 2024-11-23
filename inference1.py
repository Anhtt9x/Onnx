import sys
import json
import cv2
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper

model = "mnist-1/mnist/model.onnx"
path = sys.argv[1]

img = cv2.imread(path)
img = np.dot(img[...,:3],[0.299, 0.587, 0.114])
img = cv2.resize(img, (28, 28),interpolation=cv2.INTER_AREA)
img.resize(1,1,28,28)

data = json.dumps({"data":img.tolist()})
data = np.array(json.loads(data)['data']).astype(np.float32)
session = onnxruntime.InferenceSession(model,None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run([output_name], {input_name: data})[0]
predict = int(np.argmax(np.array(result).squeeze(),axis=0))
print(f"Predict:{predict}")  # Output: 5