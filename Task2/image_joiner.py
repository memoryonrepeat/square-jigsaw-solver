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

Be aware that the image must be the same size in width and height.
"""

class ImageJoiner:

    def __init__(self, params):
        self.params = params
        self.MARGIN = 10 # margin size in pixel
        self.solved = []

    def compareHistogram(self, first, second, first_side, second_side):
        metrics = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
        histogram1 = self.calculateHistogram(self.getMargin(first, first_side))
        histogram2 = self.calculateHistogram(self.getMargin(second, second_side))
        return cv2.compareHist(histogram1, histogram2, metrics[0])

    def calculateHistogram(self, patch):
        hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def getMargin(self, patch, position):
        patch_size = len(patch[0])
        margin_map = {
            "left": patch[:, 0:self.MARGIN],
            "right": patch[:, patch_size - self.MARGIN : patch_size],
            "top": patch[0 : self.MARGIN, :],
            "bottom": patch[patch_size - self.MARGIN : patch_size, :],
        }

        if position not in margin_map:
            raise Exception('Invalid position')

        return margin_map[position]

    def solve(self):
        i = 1
        self.solved.append(self.patches[0])
        self.patches.pop(0)
        current = self.solved[0]
        while len(self.patches) > 0:
            if i < self.row_count and i % self.row_count != 0: # first row, only compare left margin
                print(i, "first row")
                most_fit = sorted(enumerate(self.patches), key=lambda tuple: self.compareHistogram(current, tuple[1], "right", "left"), reverse=True)[0]
            elif i % self.row_count == 0: # first column, only compare top margin to the one right above
                print(i, "first column")
                most_fit = sorted(enumerate(self.patches), key=lambda tuple: self.compareHistogram(self.solved[i-self.row_count], tuple[1], "bottom", "top"), reverse=True)[0]
            else: # the rest --> compare to right of current and bottom of the above
                print(i, "rest")
                most_fit = sorted(enumerate(self.patches), key=lambda tuple: self.compareHistogram(current, tuple[1], "right", "left") + self.compareHistogram(self.solved[i-self.row_count], tuple[1], "bottom", "top"), reverse=True)[0]
            current = most_fit[1]
            self.solved.append(current)
            self.patches.pop(most_fit[0])
            i += 1


    def run(self):
        image = cv2.imread(self.params.image)
        self.image = image

        if image.shape[0] != image.shape[1]:
            print('Height and width of the puzzle must be the same')
            exit(1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        # offset = next(i for i in range(0, image.shape[0]) if image[i][i][0] != 0 or image[i][i][1] != 0 or image[i][i][2] != 0)
        offset = next(i for i in range(0, image.shape[0]) if any(image[i][j][k] != 0 for j in range(0, image.shape[0]) for k in range(0, 3)))
        print(offset)
        self.size = offset * 2

        # offset = int(self.size / 2)
        row_count = int((image.shape[0] - offset) / self.size / 1.5)
        self.row_count = row_count

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
                output[top_point : top_point + self.size, left_point : left_point + self.size, :] = self.solved[index_count]
                left_point += self.size
                index_count += 1
            left_point = 0
            top_point += self.size

        cv2.imwrite(self.params.output, output)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="input image")
    ap.add_argument("-o", "--output", required=True, help="output path of the image")

    args = ap.parse_args()

    splitter = ImageJoiner(args)
    splitter.run()
