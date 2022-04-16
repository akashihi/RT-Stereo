import torch
import torch.nn as nn
import argparse
import sys

parser = argparse.ArgumentParser(description='GwcNet -> ONNX converter')
parser.add_argument('--net', default='./GwcNet/', help='Location of GwcNet code')
parser.add_argument('--weights', default='./sceneflow.ckpt', help='Location of pre-trained weights file')
parser.add_argument('--width', default=672, type=int, help='Input image width, should be divisible by 3 and 16')
parser.add_argument('--height', default=480, type=int, help='Input image height, should be divisible by 8')
parser.add_argument('--disparity', default=96, type=int, help='Number of disparity levels')
parser.add_argument('--output', default='model.onnx', help='Model output file name')

args = parser.parse_args()
if args.width % 3 != 0 or args.width % 16 != 0:
    raise ValueError("Width should be divisible by 3 and 16")
if args.height % 8 != 0:
    raise ValueError("Height should be divisible by 8")

sys.path.append(args.net)
from models import GwcNet_GC

model = GwcNet_GC(args.disparity).cuda()
model = nn.DataParallel(model)

state_dict = torch.load(args.weights)
model.load_state_dict(state_dict['model'])
model.eval()
with torch.no_grad():
    dummy_input_left = torch.randn(1, 3, args.height, args.width).cuda()
    dummy_input_right = torch.randn(1, 3, args.height, args.width).cuda()
    torch.onnx.export(model.module, (dummy_input_left, dummy_input_right), args.output, export_params=True,
                      opset_version=11, do_constant_folding=True, input_names=['left', 'right'],
                      output_names=['output'])
