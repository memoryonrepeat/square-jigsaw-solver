import cv2
import random
import argparse
import numpy as np

"""

Reads in n x n small images and join them based on original order
Assuming offset = size / 2

Input parameters:
    -i, --image: path to the original puzzle
    -o, --output: path where the processed image should be saved (path + filename)
    -s, --size: size of each patch

Be aware that the image must be the same size in width and height.
"""

class ImageJoiner:

    def __init__(self, params):
        self.params = params
        self.size = int(self.params.size)
        self.MARGIN = 5 # margin size in pixel

    def compareHistogram(self, first, second):
        metrics = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
        histogram1 = self.calculateHistogram(first)
        histogram2 = self.calculateHistogram(second)
        return cv2.compareHist(histogram1, histogram2, metrics[0])

    def calculateHistogram(self, patch):
        hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def getMargin(self, patch):
        left_margin = patch[:, 0:self.MARGIN]
        return left_margin

    def solve(self):
        first = self.getMargin(self.patches[0])
        second = self.getMargin(self.patches[5])
        print(self.compareHistogram(first, second))

    def run(self):
        image = cv2.imread(self.params.image)
        self.image = image

        if image.shape[0] != image.shape[1]:
            print('Height and width of the puzzle must be the same')
            exit(1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        offset = int(self.size / 2)
        row_count = int((image.shape[0] - offset) / self.size / 1.5)

        new_image_size = int(row_count * self.size)
        output = np.zeros((new_image_size, new_image_size, 4)) # 4 = number of color channels
        self.patches = [image[offset * (i+1) + i * self.size : offset * (i+1) + (i+1) * self.size, \
                 offset * (j+1) + j * self.size : offset * (j+1) + (j+1) * self.size] \
                 for i in range(0, row_count) for j in range(0, row_count)]

        self.solve()

        top_point = 0
        left_point = 0
        index_count = 0
        for column in range(0, row_count):
            for row in range(0, row_count):
                output[top_point : top_point + self.size, left_point : left_point + self.size, :] = self.patches[index_count]
                left_point += self.size
                index_count += 1
            left_point = 0
            top_point += self.size

        cv2.imwrite(self.params.output, output)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="input image")
    ap.add_argument("-o", "--output", required=True, help="output path of the image")
    ap.add_argument("-s", "--size", required=True, help="size of the cubes")

    args = ap.parse_args()

    splitter = ImageJoiner(args)
    splitter.run()
