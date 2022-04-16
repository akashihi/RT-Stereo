import onnxruntime as ort
import numpy as np
import argparse
from PIL import Image
import time
import os
import cv2

os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"

parser = argparse.ArgumentParser(description='One shot stereo depth estimation')
parser.add_argument('--model', default='model.onnx', help='Model to use')
parser.add_argument('left', help='Left stereo image')
parser.add_argument('right', help='Right stereo image')
parser.add_argument('--output', default='output.png', help='Disparity map output filename')

args = parser.parse_args()


def load_image(fname):
    im = Image.open(fname).convert('RGB')
    im = np.array(im)

    im = np.transpose(im, (2, 0, 1))
    im = (im / 255).astype(np.float32)

    # Values are taken from the GwcNet transforms definition
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im[0] = (im[0] - mean[0]) / std[0]
    im[1] = (im[1] - mean[1]) / std[1]
    im[2] = (im[2] - mean[2]) / std[2]

    im = np.expand_dims(im, axis=0)
    return im


im_left = load_image(args.left)
im_right = load_image(args.right)

print(f"Loading model from {args.model}")
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
ort_sess = ort.InferenceSession(args.model, sess_options=sess_options,
                                providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
start_time = time.time()
outputs = ort_sess.run(None, {'left': im_left, 'right': im_right})
print(f"--- {time.time() - start_time} seconds ---")
disparity = np.squeeze(outputs[0], axis=0)
disp_est_uint = np.round(disparity).astype(np.uint8)
disp_im = cv2.applyColorMap(disp_est_uint, cv2.COLORMAP_JET)
cv2.imwrite(args.output, disp_im)
