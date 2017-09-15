import sys
from PIL import Image

im = Image.open(sys.argv[1])
pixel = im.load()

for i in range(0, im.size[0]):
   for j in range(0, im.size[1]):
      pixel[i, j] = (pixel[i, j][0] // 2, pixel[i, j][1] // 2, pixel[i, j][2] // 2)

im.save("Q2.png")