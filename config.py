import os

filt_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(filt_path) + os.path.sep + ".")

# dbnet 参数
dbnet_max_size = 6000  # 长边最大长度


# crnn参数
model_path = os.path.join(father_path, "models/dbnet.onnx")
is_rgb = True
crnn_model_path = os.path.join(father_path, "models/crnn_lite_lstm.onnx")



# angle
angle_detect = True
angle_detect_num = 30
angle_net_path = os.path.join(father_path, "models/angle_net.onnx")

version = 'api/v1'
