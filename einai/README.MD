## 正名

这个一个很好玩的项目，我需要给他起一个好玩的名字，下面是一串列表，列表中的每一项都和这个项目要创造的产品有一定的关联：

Apple-Siri

Microsoft-Cortana

Interstellar-TARS

这些名字很好听，我本来想仿造一个，但是意义又不是很大，原因是每一个独自又不能构成仿造的对象，于是决定仿照Linux/GNU之类的起名方式吧，就Einai Is not AsIstant了，当然还有一些细节是留白。

## 出发点

现在，也就是2021年，手机和电脑是信息世界的主要载体。人和这些设备的交互方式主要是物理按键和触控，进一步很多人很多产品都尝试用人类自然语言完成交互，但是都不太如意，进展不大，显然这存在一种中间状态，这样就构成我的出发点:用语言来指导Einai完成按键和触控。

注：之所以会形成这样的概念，是因为在思考无人驾驶中的一些感知问题:现有的无人驾驶识别和人的识别有什么差别。我的关注点最后落在了人的识别和人与物体的交互是同时发生的而现有的识别是没有这种交互的，这种交互能带来什么呢？这是一个设计很多的问题，甚至可以拔高到象征/图像之类的思考，不展开。于是我想有没有一种识别模型，输入是一些列的图片，图片中有一些无人车需要面对的场景，一个Agent在3D引擎中可以用纯物理接触的方式把这个场景摆放出来，如果说摆放这个任务完成了，是不是完成了识别，这样定义识别是不是更加具体清晰。

## 设计

[https://excalidraw.com/#room=32928785fe5c08a59db0,wzb72Kothtj4jO3Gt-cDPg](https://excalidraw.com/#room=32928785fe5c08a59db0,wzb72Kothtj4jO3Gt-cDPg)

（TODO）

一种是基于训练的，一种是基于规则的类似快捷指令

这个安全系统怎么设计，误操作了怎么办呢？一种是对危险操作进行基于规则的界定，一种是对危险动作的设定

那么他的任务有哪些呢？

简单的语言任务（TODO）

已有的gym任务比如Atari

负责的放置任务，比如根据游戏的场景移动和布置

和人的交互怎么设计一种是通过文字or语音，一种是示教，（TODO要不要鼠标抢占，抢占了以后如何维护鼠标场景的一致性）

## 具体实现

- xdotool [https://www.semicomplete.com/projects/xdotool/](https://www.semicomplete.com/projects/xdotool/)
- 用python模拟鼠标和按键

```python
# !pip install PyUserInput
from pykeyboard import *
from pymouse import *
m = PyMouse()
k = PyKeyboard()
location1=m.position()
location2=m.position()
m.click(location1[0],location1[1])
k.type_string('I love you')
m.click(location2[0],location2[1])
```

pyautogui

[https://pyautogui.readthedocs.io/en/latest/](https://pyautogui.readthedocs.io/en/latest/)

### 获取鼠标周围的截图

PIL/pyautogui

```python
import time

import numpy as np
from PIL import ImageGrab
 
img = ImageGrab.grab(bbox=(100, 161, 1141, 610))
img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

import pyautogui
 
img = pyautogui.screenshot(region=[0,0,100,100]) # x,y,w,h
img.save('screenshot.png')
```

### Text界面

```python
//Server
import socket
host = socket.gethostname()
port = 12345
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((host,port))
s.listen(1)
sock,addr = s.accept()
print('Connection built')
info = sock.recv(1024).decode()
while info != 'exit':
  print('MOOD:'+info)
  send_mes = input()
  sock.send(send_mes.encode())
  if send_mes =='exit':
    break
  info = sock.recv(1024).decode()
sock.close()
s.close()

//Client
import socket
s= socket.socket()
host = socket.gethostname()
port = 12345
s.connect((host,port))
print('Linked')
info = ''
while info != 'exit':
  print('SCIENCE:'+info)
  send_mes=input()
  s.send(send_mes.encode())
  if send_mes =='exit':
    break
  info = s.recv(1024).decode()
s.close()import socket
s= socket.socket()
host = socket.gethostname()
port = 12345
s.connect((host,port))
print('Linked')
info = ''
while info != 'exit':
  print('SCIENCE:'+info)
  send_mes=input()
  s.send(send_mes.encode())
  if send_mes =='exit':
    break
  info = s.recv(1024).decode()
s.close()
```

Text界面的调试指令，比如进入调试模式，进入训练模式等等？如何设计

### GYM环境

[https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)

一种是全部的都是连续的环境

一种是移动是连续的，click和type是离散的（TODO：混合的action我并没有找到例子）

### Task定制

如何生成很多条指令呢？同时我也可以和agent进行进一步的交互

method 1 采用shared automoy的环境完成相关游戏

method 2 通过xdotool的工具的搜索功能完成一些鼠标的动作+录屏

mac automator

windows loadrunner

[https://github.com/RMPR/atbswp](https://github.com/RMPR/atbswp)

method 3 参考[https://microsoft.github.io/task_oriented_dialogue_as_dataflow_synthesis/](https://microsoft.github.io/task_oriented_dialogue_as_dataflow_synthesis/)

method 4 直接去执行[https://github.com/facebookresearch/droidlet](https://github.com/facebookresearch/droidlet)的任务

method 5 图片描述任务

[https://github.com/alexandre01/deepsvg](https://github.com/alexandre01/deepsvg)

[https://github.com/googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)

method 5-2 写汉字

[http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html)

[http://web.mit.edu/cocosci/josh.html](http://web.mit.edu/cocosci/josh.html)  omniglot

method 6 简笔画任务参考openai的东西

method 7 场景还原任务

method 8 [https://github.com/facebookresearch/clevr-iep](https://github.com/facebookresearch/clevr-iep) 拆解这个数据库/Neural Task Programming

method 9 天际线游戏

对于视频的处理可以参考FLARE最新的东西

### 语言模型

中文的简单pretrain

悬浮球

[https://github.com/HalfStackDeveloper/FloatBall](https://github.com/HalfStackDeveloper/FloatBall)

[https://github.com/hanbaokun](https://github.com/hanbaokun)

## 模型初步结构

采用层级控制的结构，High level是Discrete，low level是Continuous。

初步会用MCTS/PPO/CLIP相关的东西

## 语音相关

语音识别和语音合成

[https://github.com/lturing/tacotronv2_wavernn_chinese](https://github.com/lturing/tacotronv2_wavernn_chinese)