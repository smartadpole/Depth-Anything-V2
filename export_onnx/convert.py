import cv2
import torch
import numpy as np
import os
import time
import argparse
from utils.file import MkdirSimple
from utils.file import ReadImageList
from depth_anything_v2.dpt import DepthAnythingV2
from export_onnx.onnx_test import test_dir

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

W = 644 # // 14
H = 392 # // 14

def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help="Device to run the model on.")
    parser.add_argument("--width", type=int, default=W, help="Width of the input image.")
    parser.add_argument("--height", type=int, default=H, help="Height of the input image.")
    parser.add_argument("--test", action="store_true", help="test model")
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

def main():
    args = parse_args()
    model_name = "_".join(args.model.split("/")[-3:]).replace("ckpts", "").replace("=", "-").strip('_')
    model_name = os.path.splitext(model_name)[0].split("-val")[0]
    output = os.path.join(args.output, model_name, f'{args.width}_{args.height}')
    onnx_file = os.path.join(output, f'DepthAnythingV2_{args.width}_{args.height}_{model_name}_12.onnx')
    MkdirSimple(output)

    export_to_onnx(args.model, onnx_file, args.width, args.height, args.device)  # Replace 'vitl' with the desired encoder

    print("export onnx to {}".format(onnx_file))
    if args.test:
        test_dir(onnx_file, [args.image, ], output)


if __name__ == "__main__":
    main()
