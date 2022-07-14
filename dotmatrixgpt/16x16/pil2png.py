# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageFont, ImageDraw

def text2pil(text):
    # text = u"这是一段测试文本，test 123。"
    im = Image.new("RGB", (32, 32), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    # 寒蝉宽黑体.otf 13 (0, 1) 寒蝉点阵体7px.ttf 凤凰点阵体16px.ttf 
    # 寒蝉手拙体2.0.ttf 20 (-1, -3)  莫妮卡x12y16px.otf
    font = ImageFont.truetype(os.path.join("寒蝉手拙体2.0.ttf"), 40)
    dr.text((0, -6), text, font=font, fill="#000000")
    im.show()
    im.save("t.png")

text2pil("道")
