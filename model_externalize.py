import onnx
import argparse

parser = argparse.ArgumentParser(description='Resaves ONNX model in external storage format')
parser.add_argument('--input', default='model.onnx', help='Model to convert')
parser.add_argument('--output', default='model_external.onnx', help='Target file name')

args = parser.parse_args()

onnx_model = onnx.load(args.input)
onnx.save_model(onnx_model, args.output, location=f"{args.output}.ext", save_as_external_data=True, all_tensors_to_one_file=True, size_threshold=1024, convert_attribute=False)
