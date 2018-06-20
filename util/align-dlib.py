#!/usr/bin/env python2


import argparse
import cv2
import numpy as np
import os
import random
import shutil
import csv
import subprocess
import requests
from google_images_download import google_images_download
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import json



import openface
import openface.helper
from openface.data import iterImgs

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
current_dir = os.getcwd()
# funcktion that searches the web and downloads images of persons mentioned in /util/training-images.csv
def download_p():

    response = google_images_download.googleimagesdownload()
    arguments = []

    with open("util/training-images.csv") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for r in reader:
            p = os.path.isdir(current_dir+'/training-images/'+r[0])
            if p:
                continue
            arguments.append({
                "keywords": r[0],
                "limit": r[1],
                "format": "jpg",
                "output_directory": current_dir + "/training-images"})

    for argument in arguments:
        response.download(argument)

    print ' FINISHED DOWNLOADING '

# write data to csv
def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")


def computeMeanMain(args):
    align = openface.AlignDlib(args.dlibFacePredictor)

    imgs = list(iterImgs(args.inputDir))

    facePoints = []
    for img in imgs:
        rgb = img.getRGB()
        bb = align.getLargestFaceBoundingBox(rgb)
        alignedPoints = align.align(rgb, bb)
        if alignedPoints:
            facePoints.append(alignedPoints)

    facePointsNp = np.array(facePoints)
    mean = np.mean(facePointsNp, axis=0)
    std = np.std(facePointsNp, axis=0)

    # Only import in this mode.
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(mean[:, 0], -mean[:, 1], color='k')
    ax.axis('equal')
    for i, p in enumerate(mean):
        ax.annotate(str(i), (p[0] + 0.005, -p[1] + 0.005), fontsize=8)
    plt.savefig("{}/mean.png".format(args.modelDir))


def alignMain(args):
    openface.helper.mkdirP(args.outputDir)
    # nei represents folders that have less than 10 images that moved to NEI folder to be updated properly
    num_nei = 0
    folders = ([name for name in os.listdir(args.inputDir)
                if os.path.isdir(os.path.join(args.inputDir, name))])
    for folder in folders:
        contents = os.listdir(os.path.join(args.inputDir, folder))  # get list of contents
        if len(contents) < args.numImages:
            shutil.move(current_dir + "/training-images/"+folder, current_dir + "/nei")
            num_nei+=1
    if(num_nei > 0):
        print("{} folders with less than 10 images moved to 'nei' folder".format(num_nei))

    imgs = list(iterImgs(args.inputDir))
    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    if args.landmarks not in landmarkMap:
        raise Exception("Landmarks unrecognized: {}".format(args.landmarks))

    landmarkIndices = landmarkMap[args.landmarks]

    align = openface.AlignDlib(args.dlibFacePredictor)

    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(args.outputDir, imgObject.cls)
        openface.helper.mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"

        if os.path.isfile(imgName):
            if args.verbose:
                print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                outRgb = None
            else:
                outRgb = align.align(args.size, rgb,
                                     landmarkIndices=landmarkIndices,
                                     skipMulti=args.skipMulti)
                if outRgb is None and args.verbose:
                    print("  + Unable to align.")

            if outRgb is not None:
                # continue
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)
                # print(imgObject.name)
            # else:
            #     shutil.move(imgObject.path, os.path.normpath(os.getcwd() + os.sep + "img_no_face/" + imgObject.name +"jpg"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('inputDir', type=str, help="Input image directory.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    computeMeanParser = subparsers.add_parser(
        'computeMean', help='Compute the image mean of a directory of images.')
    computeMeanParser.add_argument('--numImages', type=int, help="The number of images. '0' for all images.",
                                   default=10)  # <= 0 ===> all imgs
    alignmentParser = subparsers.add_parser(
        'align', help='Align a directory of images.')
    alignmentParser.add_argument('landmarks', type=str,
                                 choices=['outerEyesAndNose',
                                          'innerEyesAndBottomLip',
                                          'eyes_1'],
                                 help='The landmarks to align to.')
    alignmentParser.add_argument('--numImages', type=int, help="If the number of images in trainig folder are less than 10 the folder will be moved to nei",
                                   default=10)
    alignmentParser.add_argument(
        'outputDir', type=str, help="Output directory of aligned images.")
    alignmentParser.add_argument('--size', type=int, help="Default image size.",
                                 default=96)
    alignmentParser.add_argument('--fallbackLfw', type=str,
                                 help="If alignment doesn't work, fallback to copying the deep funneled version from this directory..")
    alignmentParser.add_argument(
        '--skipMulti', action='store_true', help="Skip images with more than one face.")
    alignmentParser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    #download_p()
    # es = Elasticsearch()
    # es.create(index='111', doc_type='doc', id='333', body = {'baby':'baby'} )
    # data = es.get(index='111', doc_type='doc', id='333',ignore='404')
    # print(json.dumps((data),indent=4))

    if args.mode == 'computeMean':
        computeMeanMain(args)
    else:
        alignMain(args)
