#!/usr/bin/env python

import cv2
import depthai as dai
import numpy as np
import os
import onnxruntime as ort
import argparse

os.environ['ORT_TENSORRT_FP16_ENABLE'] = '1'
os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'
os.environ['ORT_TENSORRT_CACHE_PATH'] = '/tmp'

parser = argparse.ArgumentParser(description='Real time stereo depth estimation')
parser.add_argument('--model', default='model.onnx', help='Model to use')
parser.add_argument('--output_prefix',
                    help='Writes disparity maps to the file <prefix>_XXXXX.png instead of screen output')

args = parser.parse_args()
print(args)


def getMesh(calibData):
    resolution = (1280, 720)
    M1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, resolution[0], resolution[1]))
    d1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
    R1 = np.array(calibData.getStereoLeftRectificationRotation())
    M2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, resolution[0], resolution[1]))
    d2 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
    R2 = np.array(calibData.getStereoRightRectificationRotation())
    mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, M2, resolution, cv2.CV_32FC1)
    mapXR, mapYR = cv2.initUndistortRectifyMap(M2, d2, R2, M2, resolution, cv2.CV_32FC1)

    meshCellSize = 16
    meshLeft = []
    meshRight = []

    for y in range(mapXL.shape[0] + 1):
        if y % meshCellSize == 0:
            rowLeft = []
            rowRight = []
            for x in range(mapXL.shape[1] + 1):
                if x % meshCellSize == 0:
                    if y == mapXL.shape[0] and x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y - 1, x - 1])
                        rowLeft.append(mapXL[y - 1, x - 1])
                        rowRight.append(mapYR[y - 1, x - 1])
                        rowRight.append(mapXR[y - 1, x - 1])
                    elif y == mapXL.shape[0]:
                        rowLeft.append(mapYL[y - 1, x])
                        rowLeft.append(mapXL[y - 1, x])
                        rowRight.append(mapYR[y - 1, x])
                        rowRight.append(mapXR[y - 1, x])
                    elif x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y, x - 1])
                        rowLeft.append(mapXL[y, x - 1])
                        rowRight.append(mapYR[y, x - 1])
                        rowRight.append(mapXR[y, x - 1])
                    else:
                        rowLeft.append(mapYL[y, x])
                        rowLeft.append(mapXL[y, x])
                        rowRight.append(mapYR[y, x])
                        rowRight.append(mapXR[y, x])
            if (mapXL.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)
                rowRight.append(0)
                rowRight.append(0)

            meshLeft.append(rowLeft)
            meshRight.append(rowRight)

    meshLeft = np.array(meshLeft)
    meshRight = np.array(meshRight)

    return meshLeft, meshRight


def process_image(im):
    im = cv2.resize(im, (816, 480), interpolation=cv2.INTER_AREA)
    im = im[0:480, 72:744]

    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

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


# Camera input pipeline
pipeline = dai.Pipeline()

camLeft = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xoutRectifLeft = pipeline.create(dai.node.XLinkOut)
xoutRectifRight = pipeline.create(dai.node.XLinkOut)

camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

xoutRectifLeft.setStreamName("rectifiedLeft")
xoutRectifRight.setStreamName("rectifiedRight")

camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)
stereo.rectifiedLeft.link(xoutRectifLeft.input)
stereo.rectifiedRight.link(xoutRectifRight.input)

calibData = dai.Device().readCalibration()
leftMesh, rightMesh = getMesh(calibData)
stereo.loadMeshData(list(leftMesh.tobytes()), list(rightMesh.tobytes()))

ort_sess = ort.InferenceSession(args.model, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

with dai.Device(pipeline, usb2Mode=True) as device:
    leftQ = device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
    rightQ = device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)
    frame_counter = 0
    while True:
        has_left = False
        has_right = False
        left = []
        right = []
        if leftQ.has():
            left = leftQ.get().getCvFrame()
            has_left = True
        if rightQ.has():
            right = rightQ.get().getCvFrame()
            has_right = True
        if has_left and has_right:
            has_left = False
            has_right = False
            im_left = process_image(left)
            im_right = process_image(right)
            outputs = ort_sess.run(None, {'left': im_left, 'right': im_right})
            disparity = np.squeeze(outputs[0], axis=0)
            disp_est_uint = np.round(disparity).astype(np.uint8)
            disp_im = cv2.applyColorMap(disp_est_uint, cv2.COLORMAP_JET)
            if not args.output_prefix:
                cv2.imshow("disparity", disp_im)
            else:
                fname = f"{args.output_prefix}_{frame_counter:06}.png"
                cv2.imwrite(fname, disp_im)
            frame_counter += 1
        if cv2.waitKey(1) == ord("q"):
            break
