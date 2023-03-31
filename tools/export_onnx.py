import cv2
import numpy as np
import onnx
import torch.cuda
from onnxsim import simplify

from visual.model.backbone.resnet import resnet18

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes=10, include_top=True).to(devices)
model.load_state_dict(torch.load(r"D:/002 Projects/001 Python/Learn-visual-tasks/model_080.pth", map_location=devices))

inputs = torch.rand(1, 3, 224, 224).to(devices)
torch.onnx.export(model, inputs, r"D:/test.onnx", opset_version=11, input_names=["input"], output_names=["output"])

onnx_model = onnx.load("D:/test.onnx")  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, "D:/test.onnx")
print('finished exporting onnx')



