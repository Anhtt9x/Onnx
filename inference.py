import onnxruntime as rt
import numpy as np

data = np.random.rand(3,4).astype(np.float32)

sess = rt.InferenceSession("output/model.onnx")

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

pred_onnx = sess.run([output_name], {input_name: data})[0]
print(pred_onnx)
