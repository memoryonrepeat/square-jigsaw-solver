import cv2
import random
import argparse
import numpy as np
"""

Reads in n x n small images and join them based on original order
Assuming offset = int(size / 2)
The solver follows a greedy algorithm, starting from the top left patch (more details below)

1/ Why this is being used:
The fact that top left patch is at the right position is important, because that can
be used to anchor later decisions and narrow down the search space. So I decided to use greedy
approach to reduce complexity, as for each step only comparison to at most 2 previous patches was
needed. The downside is that there is no verification / backtracking yet - I didn't implement this due to increased complexity)
so if one patch is selected wrongly, it will likely snowball and lead to some zone being misplaced. 
However patches within each local areas has high chance to be placed correctly, as histogram comparison
is done between neighbors only.

2/ Time and space complexity:
Let N be the number of patches
Python internally uses Timsort, which takes O(NlogN) on average
Parsing patches takes O(N) --> Negligible
Sorting needs to be done on the array of size N -> N-1 -> ... -> 2 -> 1
--> total size to be sorted = sum of the array above = N(N+1)/2
--> Total time complexity = N(N+1)/2 * O(NlogN) ~ N^2 * O(NlogN)
Space complexity: O(2N) ~ O(N) (only one more array of patches of the same size is needed)

Input parameters:
    -i, --image: path to the original puzzle
    -o, --output: path where the processed image should be saved (path + filename)
"""

class ImageJoiner:

    def __init__(self, params):
        self.params = params
        self.MARGIN = 10 # margin size in pixel
        self.solved = [] # array of patches in solved order

    # Calculate histogram correlation score between 2 selected margins of 2 images
    # For HISTCMP_CORREL and HISTCMP_INTERSECT, the higher the better
    # For HISTCMP_CHISQR and HISTCMP_BHATTACHARYYA, the lower the better
    def compareHistogram(self, first, second, first_side, second_side):
        metrics = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
        histogram1 = self.calculateHistogram(self.getMargin(first, first_side))
        histogram2 = self.calculateHistogram(self.getMargin(second, second_side))
        return cv2.compareHist(histogram1, histogram2, metrics[0])

    # Calculate histogram of a piece of image
    def calculateHistogram(self, patch):
        hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    # Returns the cropped margin of the image, to compare later
    # Position parameter determines which of 4 sides is taken
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

    # Greedy algorithm: Select the next best fit, based on margin similarity
    # Starting from the top left, call it current
    # Find its neighbor by going through the remaining patches 
    # and select the one whose margin has highest histogram correlation to current one
    # If current one is on first row / first column, only compare left - right or top - bottom margins respectively
    # Else, compare the sum of both (since the neighbor has 2 borders instead of one)
    def solve(self):
        i = 1
        self.solved.append(self.patches[0])
        self.patches.pop(0)
        current = self.solved[0]
        while len(self.patches) > 0:
            if i < self.row_count and i % self.row_count != 0: # first row, only compare right margin of current to left margin of candidate
                most_fit = sorted(enumerate(self.patches), key=lambda tuple: self.compareHistogram(current, tuple[1], "right", "left"), reverse=True)[0]
            elif i % self.row_count == 0: # first column, only compare bottom margin of the solved one above to top margin of candidate
                most_fit = sorted(enumerate(self.patches), key=lambda tuple: self.compareHistogram(self.solved[i-self.row_count], tuple[1], "bottom", "top"), reverse=True)[0]
            else: # the rest --> compare bottom and right margins of solved ones to top and left margins of candidate
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
        offset = next(i for i in range(0, image.shape[0]) if any(image[i][j][k] != 0 for j in range(0, image.shape[0]) for k in range(0, 3)))

        # assuming offset =  int( 1/2 patch size ), calculate correct patch size from offset
        # need to be precise because patch size can be also be 2 * offset + 1
        row_count = int((image.shape[0]/offset -1)/3)
        self.size = int((image.shape[0] - offset * (row_count + 1)) / row_count)
        self.row_count = row_count

        new_image_size = int(row_count * self.size)
        output = np.zeros((new_image_size, new_image_size, 4)) # 4 = number of color channels
        # Extract original patches
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
