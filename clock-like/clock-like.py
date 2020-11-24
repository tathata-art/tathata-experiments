#!/usr/bin/env python3

import argparse
from glob import glob
from operator import itemgetter
import os
import pickle

import numpy as np
import requests
from skimage.io import imread, imsave
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from tqdm import tqdm


IMAGE_FILENAME = 'val_256.tar'
IMAGE_DIR = 'images'


def download_images():
    print('>> download_images() called.')

    if not os.path.exists(IMAGE_FILENAME):

        # https://stackoverflow.com/a/37573701
        url = 'http://data.csail.mit.edu/places/places365/val_256.tar'
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(IMAGE_FILENAME, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")


    if not os.path.exists(IMAGE_DIR):
        cmd = f'''
        tar -xf {IMAGE_FILENAME};
        mv val_256 {IMAGE_DIR}
        '''
        os.system(cmd)

def compute_scores():
    print('>> compute_scores() called.')

    hands = {}
    for fname in glob('hands/*png'):
        hand = imread(fname)
        hand = rgb2gray(hand)
        hand = invert(hand)
        hands[fname] = hand

    def compute_similarity(hand, img):
        return ((hand - 0.01) * img).sum()

    scores = {}

    for fname in tqdm(glob(f'{IMAGE_DIR}/*jpg')):
        image = rgb2gray(imread(fname))

        for hand_key in hands.keys():
            if hand_key not in scores:
                scores[hand_key] = []

            score = compute_similarity(hands[hand_key], image)
            scores[hand_key].append((fname, score))

    for key in scores:
        scores[key] = sorted(scores[key], key=itemgetter(1), reverse=True)

    with open('scores.pkl', 'wb') as f:
        pickle.dump(scores, f)

def make_hourly_images():
    print('>> make_hourly_images() called.')

    cmd = 'mkdir -p hourly'
    os.system(cmd)

    scores = None
    with open('scores.pkl', 'rb') as f:
        scores = pickle.load(f)

    n_layered = 20
    for i in tqdm(range(12)):
        hour = '{:02d}'.format(i)
        key = f'hands/hands-{hour}.png'

        layered = np.zeros((256, 256), dtype=np.float64)
        candidates = [fname for fname, _ in scores[key][10:n_layered]]
        for i, fname in enumerate(candidates):
            image = rgb2gray(imread(fname))
            layered = 0.8 * layered + 0.2 * image

        layered = rescale_intensity(layered)
        imsave(f'hourly/{hour}.png', layered)

def make_minutely_images():
    print('>> make_minutely_images() called.')

    cmd = 'mkdir -p minutely'
    os.system(cmd)

    scores = None
    with open('scores.pkl', 'rb') as f:
        scores = pickle.load(f)

    for hour in range(12):
        for minute in range(60):
            print('>> processing {:02}:{:02} ..'.format(hour, minute))

            ratio = 1 - (float(minute) / 60)

            layered = np.zeros((256, 256), dtype=np.float64)
            image1 = rgb2gray(imread('hourly/{:02d}.png'.format(hour)))
            image2 = rgb2gray(imread('hourly/{:02d}.png'.format((hour + 1) % 12)))

            layered = 0.8 * layered + 0.2 * ratio * image1
            layered = 0.8 * layered + 0.2 * (1 - ratio) * image2

            layered = rescale_intensity(layered)
            imsave('minutely/{:02}{:02}.png'.format(hour, minute), layered)

def make_animated_output():
    print('>> make_animated_output() called.')

    cmd = 'convert minutely/*png clock-like-output.gif'
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clock-like image generator')
    parser.add_argument(
        'command', type=str,
        choices=[
            'download-images',
            'compute-scores',
            'make-hourly-images',
            'make-minutely-images',
            'make-animated-output',
        ])
    args = parser.parse_args()

    if args.command == 'download-images':
        download_images()
    elif args.command == 'compute-scores':
        compute_scores()
    elif args.command == 'make-hourly-images':
        make_hourly_images()
    elif args.command == 'make-minutely-images':
        make_minutely_images()
    elif args.command == 'make-animated-output':
        make_animated_output()
    else:
        print(f'Command "{args.command}" not supported')
