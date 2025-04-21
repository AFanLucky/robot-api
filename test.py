from PIL import Image,ImageDraw
import numpy as np

def edge(img):
    # print(img)
    np_data = np.array(img)
    # print(np_data)
    rr = np.where(np_data[:, :, 3] != 0)
    # print(type(rr))
    xmin = min(rr[1])
    ymin = min(rr[0])
    xmax = max(rr[1])
    ymax = max(rr[0])
    return xmin, ymin, xmax, ymax

img = Image.open('./saved_images/slide.png')
img = img.convert('RGBA')
img_edge = edge(img)
draw = ImageDraw.Draw(img)
draw.rectangle(img_edge, outline="red")
img.show()

print(img)