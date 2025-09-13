from ultralytics import YOLO

model = YOLO('yolo11l_train_8_best.pt')

model.export(format='engine', imgsz=1280, half=True)
# import torch
# import tensorrt as trt
# import onnx

# # 1. Export PyTorch model to ONNX
# model = torch.load('yolo11l_train_8_best.pt', weights_only=False)['model'].float()
# model.eval()

# dummy_input = torch.randn(1, 3, 1280, 1280)  # Adjust input size as needed
# torch.onnx.export(
#     model, 
#     dummy_input, 
#     "yolo11l_train_8_best.onnx",
#     export_params=True,
#     opset_version=11,
#     do_constant_folding=True,
#     input_names=['input'],
#     output_names=['output'],
#     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
# )

# # 2. Convert ONNX to TensorRT
# def build_engine(onnx_file_path, engine_file_path, max_batch_size=1):
#     logger = trt.Logger(trt.Logger.WARNING)
#     builder = trt.Builder(logger)
#     network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#     parser = trt.OnnxParser(network, logger)
    
#     with open(onnx_file_path, 'rb') as model:
#         parser.parse(model.read())
    
#     config = builder.create_builder_config()
#     config.max_workspace_size = 1 << 28  # 256MB
#     config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 for better performance
    
#     engine = builder.build_engine(network, config)
    
#     with open(engine_file_path, "wb") as f:
#         f.write(engine.serialize())
    
#     return engine

# build_engine("yolo11l_train_8_best.onnx", "yolo11l_train_8_best.trt")