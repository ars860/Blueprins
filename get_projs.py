import itertools
import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image

from dataset import get_dataloaders_unsupervised


class RectsBank:
    # nms provider
    def __init__(self, intersec_lim=0.2):
        self.rects = []
        self.limit = intersec_lim

    def intersects(self, bbox):
        for bb in self.rects:
            if self._intersection(bb, bbox) > self.limit:
                return True
        return False

    def add(self, bbox):
        self.rects.append(bbox)

    def _intersection(self, b1, b2):
        xa1, ya1, wa, ha = b1
        xb1, yb1, wb, hb = b2
        xa2, ya2 = xa1 + wa, ya1 + ha
        xb2, yb2 = xb1 + wb, yb1 + hb

        # i stands for intersection
        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)

        wi = xi2 - xi1
        hi = yi2 - yi1

        if wi <= 0 or hi <= 0:
            return 0.

        area = wi * hi
        minArea = min(wa * ha, wb * hb)
        return area / minArea


def get_white_area(thresh_img):
    # finding conturs
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # find inner white area
    get_area = lambda c: -c[2] * c[3]
    contour_sizes = [(get_area(cv2.boundingRect(contour)), contour) for contour in contours]
    sorted_contours = sorted(contour_sizes, key=lambda x: x[0])

    mask = np.zeros(thresh_img.shape, dtype=np.uint8)
    img_area = thresh_img.shape[0] * thresh_img.shape[1]
    prev_contour = sorted_contours[0][1]
    for c in sorted_contours:
        if -c[0] / img_area < 0.5:
            crop_contour = prev_contour
            cv2.drawContours(mask, [crop_contour], -1, (255, 255, 255), cv2.FILLED)
            thresh_img = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
            thresh_img[mask == 0] = 255
            return thresh_img
        elif -cv2.contourArea(c[1]) / c[0] < 0.5:
            continue
        else:
            prev_contour = c[1]


def find_conturs(cv_img):
    # return list of conturs bboxes in format [((x, y), (x1, y1)), ... ]

    # binarization
    _, thresh_img = cv2.threshold(cv_img, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

    # make contours thicker
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(thresh_img, contours, -1, 0, 3)
    # get only inner area
    thresh_img = get_white_area(thresh_img)

    # finding contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = [cv2.boundingRect(c) for c in contours]
    top_bboxes = sorted(bboxes, key=lambda b: -b[2] * b[3])

    rects_drawn = RectsBank()
    bboxes = []
    clim = 5
    c = 0
    b = 70  # border size in pixels
    for (x, y, w, h) in top_bboxes:
        if 0.6 > w * h / cv_img.size > 0.03:
            # some kind of nms
            if rects_drawn.intersects((x, y, w, h)):
                # do not need submodules
                continue
            c += 1
            rects_drawn.add((x, y, w, h))
            bbox = ((x - b, y - b), (x + w + b, y + h + b))
            bboxes.append(bbox)

        if c >= clim:
            break
    return bboxes


def crop_conturs(cv_img):
    cv_img = cv2.threshold(cv_img, 200, 255, cv2.THRESH_BINARY)[1]
    # returns list of detail images
    details = []
    bboxes = find_conturs(cv_img)
    for c, bbox in enumerate(bboxes):
        (x, y), (x1, y1) = bbox
        detail = cv_img[y:y1, x:x1]
        if detail.size > 0:
            details.append(detail)
    return details


def draw_conturs(cv_img):
    # draws bboxes on image's details
    bboxes = find_conturs(cv_img)
    for c, bbox in enumerate(bboxes):
        (x, y), (x1, y1) = bbox
        cv2.putText(cv_img, str(c), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(cv_img, (x, y), (x1, y1), 128, 5)
    return cv_img


def get_projections(img):
    """img is a grayscale image"""
    return crop_conturs(img)


def get_all_projs(path=Path() / 'xpc_11' / 'projs'):
    dataset, dataloader = get_dataloaders_unsupervised(fraction=1.0)

    cnt = 0
    for img in dataloader:
        projs = get_projections((img.numpy().squeeze() * 255.0).astype(np.uint8))
        for proj in projs:
            Image.fromarray(proj).save(path / f'{cnt}.png')
            cnt += 1


def detect_pillars(path=Path() / 'xpc_11' / 'projs', ratio=0.05, clear=False, show=False):
    for file_name in path.glob('*'):
        with Image.open(file_name) as img:
            w, h = img.size

            if w / h <= ratio or h / w <= ratio:
                if show:
                    img.show()

        if w / h <= ratio or h / w <= ratio:
            if clear:
                os.remove(Path() / file_name)


if __name__ == '__main__':
    # get_all_projs()
    # detect_pillars(ratio=1/6, clear=True, show=False)
    dataset, dataloader = get_dataloaders_unsupervised(image_folder='projs', file_format='png', fraction=1.0)

    samples = list(itertools.islice((iter(dataloader)), 10))

    for img in samples:
        plt.imshow(img.squeeze())
        plt.show()

