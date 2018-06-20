#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import imutils as imutils

start = time.time()

import argparse
import cv2
import os
import pickle
import sys
import glob
from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
current_dir = os.getcwd()


def getRep(imgPath, multiple=False):
    skip = 0
    start = time.time()
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    start = time.time()

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        print("Unable to find a face: {}".format(imgPath))
        skip = 1


    reps = []
    for bb in bbs:
        if skip==1:
            continue
        alignedFace = align.align(
            args.imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))

        rep = net.forward(alignedFace)
        reps.append((bb, rep))

    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def train(args):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(args.workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))
    # we will use the support vector classifier class from svm library
    if args.classifier == 'LinearSvm':
        clf = SVC(C=0.5, kernel='linear', probability=True)
    # The fit method of SVC class is called to train the algorithm on the training data, embeddings as embeddings and labelsNum as to whom each embedding belongs
    clf.fit(embeddings, labelsNum)
    # save the classifier as classifier.pkl
    fName = "{}/classifier.pkl".format(args.workDir)
    print("Saving classifier to '{}'".format(fName))
    print args.classifier
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)


def infer(args, multiple=False):
    with open(args.classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')
    imgs = glob.glob(args.imgs[0]+'/*.jpg')
    for img in imgs:
        counter = 0
        image = cv2.imread(img)
        print("\n=== {} ===".format(img))
        reps = getRep(img, multiple)
        for r in reps:
            rep = r[1].reshape(1, -1)
            bb = r[0]
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]

            if confidence > 0.21:

                if multiple:
                    print("Predict {} Number: {} Location:  {} with {:.2f} confidence.".format(person.decode('utf-8'), counter, bb, confidence))
                else:
                    print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))

                cv2.putText(image,str(counter)+''+person.decode('utf-8'), (bb.left(), bb.bottom() + 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 0, 0), 2, lineType=2)

            if confidence < 0.21:

                cv2.putText(image, str(counter), (bb.left(), bb.bottom() + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 0, 0), 2, lineType=2)
            cv2.rectangle(image, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (255, 0, 0), 2)
            counter += 1
        cv2.namedWindow("detected_faces", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("detected_faces", 1080, 720)

        cv2.imshow('detected_faces', image)
        cv2.waitKey(0)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',
                                        help="Train a new classifier.")
    trainParser.add_argument('--classifier',type=str,help='The type of classifier to use.',default='LinearSvm')
    trainParser.add_argument(
        'workDir',
        type=str,
        help="The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'.")

    inferParser = subparsers.add_parser(
        'infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('imgs', type=str, nargs='+',
                             help="Input image.")
    inferParser.add_argument('--multi', help="Infer multiple faces in image",
                             action="store_true")
    args = parser.parse_args()

#     if args.mode == 'infer' and args.classifierModel.endswith(".t7"):
#         raise Exception("""
# Torch network model passed as the classification model,
# which should be a Python pickle (.pkl)
#
# See the documentation for the distinction between the Torch
# network and classification models:
#
#         http://cmusatyalab.github.io/openface/demo-3-classifier/
#         http://cmusatyalab.github.io/openface/training-new-models/
#
# Use `--networkModel` to set a non-standard Torch network model.""")
    start = time.time()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)


    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args, args.multi)
