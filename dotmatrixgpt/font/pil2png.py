# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np

def text2pil(text):
    im = Image.new("RGB", (32, 32), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(os.path.join("寒蝉手拙体2.0.ttf"), 40)
    dr.text((0, -6), text, font=font, fill="#000000")
    return np.array(im)
    # im.show()
    # im.save("t.png")

with open("../vocab/vocab.txt") as file:
    for i, line in enumerate(file.readlines()):
        line = line.strip()
        if(len(line) == 0 or len(line) > 1):
            print("bad line:" + str(i))
            print("bad line:" + line)
            continue
        text2pil(line)
