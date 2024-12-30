import cv2
import torch
import numpy as np
import os
import time
import argparse
from tools.file import MkdirSimple
from tools.file import ReadImageList
from depth_anything_v2.dpt import DepthAnythingV2
from onnxmodel import ONNXModel

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

W = 644
H = 392

def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help="Device to run the model on.")
    parser.add_argument("--width", type=int, default=W, help="Width of the input image.")
    parser.add_argument("--height", type=int, default=H, help="Height of the input image.")
    return parser.parse_args()


def load_model(model_path, encoder, device):
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device).eval()
    return model


def export_to_onnx(model_path, onnx_file, width=W, height=H, device="cuda"):
    encoder = os.path.splitext(os.path.basename(model_path))[0].split('_')[-1]
    model = load_model(model_path, encoder, device)

    # Create dummy input for the model
    dummy_input = torch.randn(1, 3, height, width).to(device)  # Adjust the size as needed
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device {param.device}")
        break
    print(f"dummy_input is on device {dummy_input.device}")
    # Export the model
    torch.onnx.export(model, dummy_input, onnx_file,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True)

    print(f"Model exported to {onnx_file}")


def test_onnx(img_path, model, width=W, height=H):
    img_org = cv2.imread(img_path)
    img = cv2.resize(img_org, (width, height), cv2.INTER_LANCZOS4)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = np.subtract(img, mean)
    img = np.divide(img, std)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype("float32")

    start_time = time.time()
    output = model.forward(img)
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
    dis_array = output[0][0]
    dis_array = (dis_array - dis_array.min()) / (dis_array.max() - dis_array.min()) * 255.0
    dis_array = dis_array.astype("uint8")

    depth = cv2.resize(dis_array, (img_org.shape[1], img_org.shape[0]))
    depth = cv2.applyColorMap(cv2.convertScaleAbs(depth, 1), cv2.COLORMAP_PARULA)
    combined_img = np.vstack((img_org, depth))

    return combined_img, depth

def test_dir(image_dir, model_file, output_dir, width, height):
    model = ONNXModel(model_file)
    img_list = ReadImageList(image_dir)
    print("test image number: ", len(img_list))
    for file in img_list:
        image, depth = test_onnx(file, model, width, height)
        depth_file = os.path.join(output_dir, 'depth', os.path.basename(file))
        concat_file = os.path.join(output_dir, 'concat', os.path.basename(file))
        MkdirSimple(depth_file)
        MkdirSimple(concat_file)
        cv2.imwrite(concat_file, image)
        cv2.imwrite(depth_file, depth)

def main():
    args = parse_args()
    output_dir = os.path.join(args.output, f"{args.width}_{args.height}")
    onnx_file = os.path.join(output_dir, os.path.splitext(os.path.basename(args.checkpoint))[0] + ".onnx")
    MkdirSimple(onnx_file)
    export_to_onnx(args.checkpoint, onnx_file, args.width, args.height, args.device)  # Replace 'vitl' with the desired encoder
    test_dir(args.image, onnx_file, output_dir, args.width, args.height)


if __name__ == "__main__":
    main()
