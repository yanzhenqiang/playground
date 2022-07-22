# -*- coding: utf-8 -*-
import binascii
KEYS = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01]

# 初始化16*16的点阵位置，需要32(16x16/8)个字节才能显示一个汉字
rect_list = [] * 16
for i in range(16):
    rect_list.append([] * 16)

# utf-8 -> gb2312
utf8 = "你"
gb2312 = utf8.encode('gb2312')
print(utf8.encode())
print(gb2312)
# 将二进制编码数据转化为十六进制数据
hex_str = binascii.b2a_hex(gb2312)
print(hex_str)
#将数据按unicode转化为字符串
result = str(hex_str, encoding='utf-8')
print(result)

#前两位区码，每一区记录94个字符, 后两位位码，是汉字在其区的位置
area = eval('0x' + result[:2]) - 0xA0
index = eval('0x' + result[2:]) - 0xA0

#汉字在HZK16中的绝对偏移位置，最后乘32是因为字库中的每个汉字字模都需要32字节
offset = (94 * (area-1) + (index-1)) * 32

font_rect = None
with open("HZK16.dat", "rb") as f:
    #找到目标汉字的偏移位置
    f.seek(offset)
    #从该字模数据中读取32字节数据
    font_rect = f.read(32)

#font_rect的长度是32，此处相当于for k in range(16)
for k in range(len(font_rect) // 2):
    row_list = rect_list[k]
    for j in range(2):
        for i in range(8):
            asc = font_rect[k * 2 + j]
            flag = asc & KEYS[i]
            #数据规则获取字模中数据添加到16行每行中16个位置处每个位置
            row_list.append(flag)

for row in rect_list:
    for i in row:
        if i:
            print('0', end=' ')
        else:
            print('.', end=' ')
    print()