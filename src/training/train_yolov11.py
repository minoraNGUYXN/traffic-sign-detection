"""
Model được train ở Kaggle: https://www.kaggle.com/code/khainguyxn/minora-object-detection-traffic-sign-detection/edit
Đây là script train được tóm tắt lại
"""

from ultralytics import YOLO
# Load a models
model = YOLO("yolo11.yaml")  # build a new model from scratch
model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

with open("/kaggle/input/traffic-signs-dataset-in-yolo-format/classes.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
    print(class_names)
# Tạo danh sách dạng {index: "classname"}
names_dict = {i: name for i, name in enumerate(class_names)}

config_content = "" # dataset root dir

with open("/kaggle/working/yolo11.yaml", "w") as fw:
  fw.write(config_content)

model.train(data="/kaggle/working/yolo11.yaml", epochs=30, single_cls=True)  # train the models
metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx")  # export the model to ONNX format