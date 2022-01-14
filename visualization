# yolov4のcfgファイルのパスからcfgファイルの内容を読み込む
with open("/content/yolov4.cfg", "r") as f:
  cfg = f.read()
  
 
#レイヤーの名前やパラメータを取得
import re

layers, param, div_param = [], [], []
param_flag = False

lines = cfg.splitlines()

for i in range(0, len(lines)):
  text = lines[i].split('#')[0]
  text = text.replace(' ', '').replace('\t', '')

  if "[" and "]" in text:
    if param_flag == True:
      param += [div_param]
      #param_flag = False
    #print(re.split('[(.*)]', text)[0])
    layers += [re.split('[(.*)]', text)[0]]
    div_param = []
    param_flag = True
  elif "" != text:
    div_param += [text]

  if i == len(lines)-1:
    param += [div_param]

#確認用
#print(layers)
#print(param)
#print(len(layers))
#print(len(param))


# cfgの内容を分かりやすい形に変える
view_cfg = ""
for i in range(0, len(layers)):
  view_cfg += ( ("%3d : " %(i-1)) + layers[i] + " = " + ", ".join(param[i]) + "\n" )
  
print(view_cfg)



#画像で使う色を決める
class Color():
  def __init__(self):
    self.black = (0, 0, 0)
    self.white = (255, 255, 255)
    self.box = self.black    #層を示す長方形のふちの色
    self.img_size_text = self.black    #画像サイズの文字の色
    self.box_text = self.black    #層の名称の文字の色
    self.box_a = self.black    #層を繋ぐ矢印の色
    self.input = (211, 211, 211)    #入力の色
    self.conv = (0, 165, 255)    #畳み込み層の色
    self.filters_text = self.black    #フィルターの数の文字の色
    self.yolo = (255, 191, 0)    # yolo層の色
    self.pool = (100, 100, 255)    # max/avgpool層の色
    self.upsample = (144, 238, 144)    # upsample層の色
    self.dropout = (238, 130, 238)    # dropout層の色
    self.connected = (150, 150, 255)    #connected層の色
    self.cost = (255, 191, 0)    # cost層の色
    self.any = self.white    #上記にない他の層の色
    self.route_c = self.black    # routeの円の色
    self.route_m = self.black    # routeのプラスマークの色
    self.route_a = self.black    # routeの矢印の色
    self.short_c = (0, 0, 255)    # shortcutの円の色
    self.short_m = self.black    # shortcutのプラスマークの色
    self.short_a = (0, 0, 255)    # shortcutの矢印の色
    self.scale_ch_c = (255, 0, 0)    # scale_channelsの円の色
    self.scale_ch_m = self.black    # scale_channelsのプラスマークの色
    self.scale_ch_a = (255, 0, 0)    # scale_channelsの矢印の色
    self.sam_c = (0, 255, 0)    # samの円の色
    self.sam_m = self.black    # samのプラスマークの色
    self.sam_a = (0, 255, 0)    # samの矢印の色
    
  

#画像を作る
import cv2
import numpy as np
from google.colab.patches import cv2_imshow    # google colaboratory用

for i in range(0, len(param[0])):
  if 'width' in param[0][i]:
    img_size = int( re.split('width=', param[0][i])[1] )    # yolov4の画像の入力サイズ
img_h, img_w = 40, 400    #作成する画像の大きさ

d_ih, d_iw = int(img_h / 4), int(img_w / 4)
#print(d_ih, d_iw)    #確認用
color = Color()

def img_in(size):
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.input, thickness=-1)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.box, thickness=1)
  cv2.putText(img, 'input', (int(1.8*d_iw), int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.box_text, lineType=cv2.LINE_AA)
  cv2.putText(img, ("%d" %img_size), (int(0.75*img_w)+10, int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.img_size_text, lineType=cv2.LINE_AA)
  cv2.arrowedLine(img, (2*d_iw, 2*d_ih), (2*d_iw, img_h), color.box_a, thickness=1, tipLength=0.2)
  return img

def img_conv(filters, size):
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.conv, thickness=-1)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.box, thickness=1)
  cv2.putText(img, 'convolutional :', (int(5.8/5*d_iw), int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.box_text, lineType=cv2.LINE_AA)
  text_w = 0.75 - 0.025 * len(list(filters))
  cv2.putText(img, filters, (int(text_w*img_w)-10, int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.filters_text, lineType=cv2.LINE_AA)
  cv2.putText(img, ("%d" %size), (int(0.75*img_w)+10, int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.img_size_text, lineType=cv2.LINE_AA)
  cv2.arrowedLine(img, (2*d_iw, 2*d_ih), (2*d_iw, img_h), color.box_a, thickness=1, tipLength=0.2)
  return img

def img_yolo():
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.yolo, thickness=-1)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.box, thickness=1)
  cv2.putText(img, 'yolo', (int(1.8*d_iw), int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.box_text, lineType=cv2.LINE_AA)
  #cv2.arrowedLine(img, (2*d_iw, 2*d_ih), (2*d_iw, img_h), color.box_a, thickness=1, tipLength=0.2)
  return img

def img_pool(type):
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.pool, thickness=-1)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.box, thickness=1)
  cv2.putText(img, type, (int(1.6*d_iw), int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.box_text, lineType=cv2.LINE_AA)
  cv2.arrowedLine(img, (2*d_iw, 2*d_ih), (2*d_iw, img_h), color.box_a, thickness=1, tipLength=0.2)
  return img

def img_upsample(size):
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.upsample, thickness=-1)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.box, thickness=1)
  cv2.putText(img, 'upsample', (int(1.6*d_iw), int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.box_text, lineType=cv2.LINE_AA)
  cv2.putText(img, ("%d" %size), (int(0.75*img_w)+10, int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.img_size_text, lineType=cv2.LINE_AA)
  cv2.arrowedLine(img, (2*d_iw, 2*d_ih), (2*d_iw, img_h), color.box_a, thickness=1, tipLength=0.2)
  return img

def img_dropout():
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.dropout, thickness=-1)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.box, thickness=1)
  cv2.putText(img, 'dropout', (int(1.6*d_iw), int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.box_text, lineType=cv2.LINE_AA)
  cv2.arrowedLine(img, (2*d_iw, 2*d_ih), (2*d_iw, img_h), color.box_a, thickness=1, tipLength=0.2)
  return img

def img_connected(out):
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.connected, thickness=-1)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.box, thickness=1)
  cv2.putText(img, 'connected :', (int(1.3*d_iw), int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.box_text, lineType=cv2.LINE_AA)
  text_w = 0.75 - 0.025 * len(list(out))
  cv2.putText(img, out, (int(text_w*img_w)-10, int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.filters_text, lineType=cv2.LINE_AA)
  cv2.arrowedLine(img, (2*d_iw, 2*d_ih), (2*d_iw, img_h), color.box_a, thickness=1, tipLength=0.2)
  return img

def img_cost():
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.cost, thickness=-1)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.box, thickness=1)
  cv2.putText(img, 'cost', (int(1.8*d_iw), int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.box_text, lineType=cv2.LINE_AA)
  #cv2.arrowedLine(img, (2*d_iw, 2*d_ih), (2*d_iw, img_h), color.box_a, thickness=1, tipLength=0.2)
  return img

def img_any(name):
  l_name = 0.5 - 0.025*int(len(list(name)) / 2)
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.any, thickness=-1)
  cv2.rectangle(img, (d_iw, 0), (3*d_iw, 2*d_ih), color.box, thickness=1)
  cv2.putText(img, name, (int(l_name*img_w)-3, int(3/8*img_h)), cv2.FONT_HERSHEY_COMPLEX, 0.5*d_ih/10, color.box_text, lineType=cv2.LINE_AA)
  cv2.arrowedLine(img, (2*d_iw, 2*d_ih), (2*d_iw, img_h), color.box_a, thickness=1, tipLength=0.2)
  return img

def img_route(img_ori, num, nums_space, connect_flag):
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  if -1 in num or len(num) >= 2:
    cv2.circle(img, (2*d_iw, d_ih), int(0.7*d_ih), color.route_c, thickness=1)
    cv2.drawMarker(img, (2*d_iw, d_ih), color.route_m, markerSize=int(0.7*d_ih)+1)
  cv2.arrowedLine(img, (2*d_iw, d_ih), (2*d_iw, img_h), color.route_a, thickness=1, tipLength=0.2)
  #cv2.arrowedLine(img, (2*d_iw, d_ih+int(0.7*d_ih)+1), (2*d_iw, img_h), color.route_a, thickness=1, tipLength=0.2)
  img = np.vstack((img_ori, img))
  h, w, c = img.shape
  cv2.arrowedLine(img, (2*d_iw, h-6*d_ih+1), (2*d_iw, h-img_h), (255, 255, 255), thickness=1, tipLength=0.2)

  if connect_flag == True:
    conct_w, conct_h = 2*d_iw, int(3/2*d_ih)
  else:
    conct_w, conct_h = d_iw, 0

  for i in range(0, len(num)):
    if num[i] < 0:
      if num[i] == -1:
        cv2.arrowedLine(img, (2*d_iw, h-img_h-2*d_ih), (2*d_iw, h-img_h), color.box_a, thickness=1, tipLength=0.2)
      else:
        if len(num) == 1:
          pts = np.array(((conct_w, h+img_h*num[i]-3*d_ih+conct_h), \
                          (d_iw-int(d_iw/10)*nums_space[i], h+img_h*num[i]-3*d_ih+conct_h), \
                          (d_iw-int(d_iw/10)*nums_space[i], h-3*d_ih), \
                          (2*d_iw, h-3*d_ih)))
          cv2.polylines(img, [pts], False, color.route_a, thickness=1)
        else:
          pts = np.array(((conct_w, h+img_h*num[i]-3*d_ih+conct_h), \
                          (d_iw-int(d_iw/10)*nums_space[i], h+img_h*num[i]-3*d_ih+conct_h), \
                          (d_iw-int(d_iw/10)*nums_space[i], h-3*d_ih), \
                          (2*d_iw-int(0.7*d_ih)-2, h-3*d_ih)))
          cv2.polylines(img, [pts], False, color.route_a, thickness=1)
          cv2.arrowedLine(img, (2*(d_iw-d_ih-1)-int(0.7*d_ih), h-3*d_ih), (2*d_iw-int(0.7*d_ih)-2, h-3*d_ih), color.route_a, thickness=1, tipLength=0.2)
    elif num[i] > 0:
      if len(num) == 1:
        pts = np.array(((conct_w, img_h*num[i]+5*d_ih+conct_h), \
                        (d_iw-int(d_iw/10)*nums_space[i], img_h*num[i]+5*d_ih+conct_h), \
                        (d_iw-int(d_iw/10)*nums_space[i], h-3*d_ih), \
                        (2*d_iw, h-3*d_ih)))
        cv2.polylines(img, [pts], False, color.route_a, thickness=1)
      else:
        pts = np.array(((conct_w, img_h*num[i]+5*d_ih+conct_h), \
                        (d_iw-int(d_iw/10)*nums_space[i], img_h*num[i]+5*d_ih+conct_h), \
                        (d_iw-int(d_iw/10)*nums_space[i], h-3*d_ih), \
                        (2*d_iw-int(0.7*d_ih)-2, h-3*d_ih)))
        cv2.polylines(img, [pts], False, color.route_a, thickness=1)
        cv2.arrowedLine(img, (2*(d_iw-d_ih-1)-int(0.7*d_ih), h-3*d_ih), (2*d_iw-int(0.7*d_ih)-2, h-3*d_ih), color.route_a, thickness=1, tipLength=0.2)
  return img

def img_shortcut(img_ori, num, connect_flag):
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.circle(img, (2*d_iw, d_ih), int(0.7*d_ih), color.short_c, thickness=1)
  cv2.drawMarker(img, (2*d_iw, d_ih), color.short_m, markerSize=int(0.7*d_ih)+1)
  cv2.arrowedLine(img, (2*d_iw, d_ih+int(0.7*d_ih)+1), (2*d_iw, img_h), color.short_a, thickness=1, tipLength=0.2)
  img = np.vstack((img_ori, img))
  h, w, c = img.shape
  cv2.arrowedLine(img, (2*d_iw, h-6*d_ih), (2*d_iw, h-img_h), color.box_a, thickness=1, tipLength=0.2)

  if connect_flag == True:
    conct_w, conct_h = 2*d_iw, h+img_h*num-int(3/2*d_ih)
  else:
    conct_w, conct_h = 3*d_iw, h+img_h*num-3*d_ih
  pts = np.array(((conct_w, conct_h), \
                  (int(0.95*img_w), conct_h), \
                  (int(0.95*img_w), h-3*d_ih), \
                  (2*d_iw+int(0.7*d_ih)+1, h-3*d_ih)))
  cv2.polylines(img, [pts], False, color.short_a, thickness=1)
  cv2.arrowedLine(img, (2*(d_iw+d_ih)+int(0.7*d_ih)+1, h-3*d_ih), (2*d_iw+int(0.7*d_ih)+1, h-3*d_ih), color.short_a, thickness=1, tipLength=0.2)
  return img

def img_scale_channels(img_ori, num, connect_flag):
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.circle(img, (2*d_iw, d_ih), int(0.7*d_ih), color.scale_ch_c, thickness=1)
  cv2.drawMarker(img, (2*d_iw, d_ih), color.scale_ch_m, markerType=cv2.MARKER_TILTED_CROSS, markerSize=int(0.7*d_ih)+1)
  cv2.arrowedLine(img, (2*d_iw, d_ih+int(0.7*d_ih)+1), (2*d_iw, img_h), color.scale_ch_a, thickness=1, tipLength=0.2)
  img = np.vstack((img_ori, img))
  h, w, c = img.shape
  cv2.arrowedLine(img, (2*d_iw, h-6*d_ih), (2*d_iw, h-img_h), color.box_a, thickness=1, tipLength=0.2)

  if connect_flag == True:
    conct_w, conct_h = 2*d_iw, h+img_h*num-int(3/2*d_ih)
  else:
    conct_w, conct_h = 3*d_iw, h+img_h*num-3*d_ih
  pts = np.array(((conct_w, conct_h), \
                  (int(0.9*img_w), conct_h), \
                  (int(0.9*img_w), h-3*d_ih), \
                  (2*d_iw+int(0.7*d_ih)+1, h-3*d_ih)))
  cv2.polylines(img, [pts], False, color.scale_ch_a, thickness=1)
  cv2.arrowedLine(img, (2*(d_iw+d_ih)+int(0.7*d_ih)+1, h-3*d_ih), (2*d_iw+int(0.7*d_ih)+1, h-3*d_ih), color.scale_ch_a, thickness=1, tipLength=0.2)
  return img

def img_sam(img_ori, num, connect_flag):
  img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
  cv2.circle(img, (2*d_iw, d_ih), int(0.7*d_ih), color.sam_c, thickness=1)
  cv2.drawMarker(img, (2*d_iw, d_ih), color.sam_m, markerType=cv2.MARKER_TILTED_CROSS, markerSize=int(0.7*d_ih)+1)
  cv2.arrowedLine(img, (2*d_iw, d_ih+int(0.7*d_ih)+1), (2*d_iw, img_h), color.sam_a, thickness=1, tipLength=0.2)
  img = np.vstack((img_ori, img))
  h, w, c = img.shape
  cv2.arrowedLine(img, (2*d_iw, h-6*d_ih), (2*d_iw, h-img_h), color.box_a, thickness=1, tipLength=0.2)

  if connect_flag == True:
    conct_w, conct_h = 2*d_iw, h+img_h*num-int(3/2*d_ih)
  else:
    conct_w, conct_h = 3*d_iw, h+img_h*num-3*d_ih
  pts = np.array(((conct_w, conct_h), \
                  (int(0.85*img_w), conct_h), \
                  (int(0.85*img_w), h-3*d_ih), \
                  (2*d_iw+int(0.7*d_ih)+1, h-3*d_ih)))
  cv2.polylines(img, [pts], False, color.sam_a, thickness=1)
  cv2.arrowedLine(img, (2*(d_iw+d_ih)+int(0.7*d_ih)+1, h-3*d_ih), (2*d_iw+int(0.7*d_ih)+1, h-3*d_ih), color.sam_a, thickness=1, tipLength=0.2)
  return img

def cal_size(size, ker, str, pad):
  return int((size - ker + 2*pad)/str) + 1



list_img_size = np.zeros(len(layers))
line_space = np.zeros((9, len(layers)-1))

list_img_size[0] = img_size
for i in range(1, len(layers)):
  if layers[i] == "[convolutional]":
    for j in range(0, len(param[i])):
      if 'filters=' in param[i][j]:
        num_param = re.split('filters=', param[i][j])[1]
      elif 'size' in param[i][j]:
        ker = int( re.split('size=', param[i][j])[1] )
      elif 'stride' in param[i][j]:
        str = int( re.split('stride=', param[i][j])[1] )
      elif 'padding' in param[i][j]:
        pad = int( re.split('padding=', param[i][j])[1] )
      elif 'pad' in param[i][j]:
        pad_f = int( re.split('pad=', param[i][j])[1] )
    if pad_f == 1:
      pad = int(ker/2)
    list_img_size[i] = cal_size(list_img_size[i-1], ker, str, pad)
    img_layer = img_conv(num_param, list_img_size[i])
  elif layers[i] == "[yolo]":
    list_img_size[i] = list_img_size[i-1]
    img_layer = img_yolo()
  elif layers[i] == "[maxpool]" or layers[i] == "[avgpool]":
    list_img_size[i] = list_img_size[i-1]
    if layers[i] == "[maxpool]":
      img_layer = img_pool('maxpool')
    if layers[i] == "[avgpool]":
      img_layer = img_pool('avgpool')
  elif layers[i] == "[upsample]":
    for j in range(0, len(param[i])):
      if 'stride' in param[i][j]:
        str = int( re.split('stride=', param[i][j])[1] )
    list_img_size[i] = list_img_size[i-1] * str
    img_layer = img_upsample(list_img_size[i])
  elif layers[i] == "[dropout]":
    list_img_size[i] = list_img_size[i-1]
    img_layer = img_dropout()
  elif layers[i] == "[connected]":
    list_img_size[i] = list_img_size[i-1]
    for j in range(0, len(param[i])):
      if 'output=' in param[i][j]:
        num_param = re.split('output=', param[i][j])[1]
    img_layer = img_connected(num_param)
  elif layers[i] == "[cost]":
    list_img_size[i] = list_img_size[i-1]
    img_layer = img_cost()
  elif layers[i] == "[route]":
    connect_flag = False
    nums_route, nums_space = [], []
    num_param = re.split('layers=', param[i][0])[1]
    num_param = num_param.split(',')

    for j in range(0, len(num_param)):
      n_r = int(num_param[j])
      nums_route += [n_r]
      if n_r != -1:
        if n_r < 0:
          list_img_size[i] = list_img_size[i+n_r]
          if layers[i+n_r] == '[shortcut]':
            connect_flag = True
          for k in range(0, 9):
            #print("k=", k+1, i+n_r, ":", i, "\nlist=", line_space[k, i+n_r:i])    #確認用
            if all(x == 0 for x in line_space[k, i+n_r:i]):
              line_space[k, i+n_r:i] = 1
              nums_space += [k+1]
              break
        elif n_r > 0:
          list_img_size[i] = list_img_size[n_r]
          if layers[n_r+1] == '[shortcut]':
            connect_flag = True
          for k in range(0, 9):
            #print("k=", k+1, n_r, ":", i, "\nlist=", line_space[k, n_r:i])    #確認用
            if all(x == 0 for x in line_space[k, n_r:i]):
              line_space[k, n_r:i] = 1
              nums_space += [k+1]
              break
      else:
        list_img_size[i] = list_img_size[i-1]
        nums_space += [0]
        
    #print(nums_route, nums_space)    #確認用
    img = img_route(img, nums_route, nums_space, connect_flag)
    continue
  elif layers[i] == "[shortcut]":
    connect_flag = False
    num_param = re.split('from=', param[i][0])[1]
    if layers[i+int(num_param)] == '[shortcut]':
      connect_flag = True
    list_img_size[i] = list_img_size[i+int(num_param)]
    img = img_shortcut(img, int(num_param), connect_flag)
    continue
  elif layers[i] == "[scale_channels]":
    connect_flag = False
    num_param = re.split('from=', param[i][0])[1]
    if layers[i+int(num_param)] == '[shortcut]':
      connect_flag = True
    list_img_size[i] = list_img_size[i+int(num_param)]
    img = img_scale_channels(img, int(num_param), connect_flag)
    continue
  elif layers[i] == "[sam]":
    connect_flag = False
    num_param = re.split('from=', param[i][0])[1]
    if layers[i+int(num_param)] == '[shortcut]':
      connect_flag = True
    list_img_size[i] = list_img_size[i+int(num_param)]
    img = img_sam(img, int(num_param), connect_flag)
    continue
  else:
    name = layers[i]
    name = name.replace('[', '').replace(']', '')
    list_img_size[i] = list_img_size[i-1]
    img_layer = img_any(name)

  if i == 1:
    img = np.vstack((img_in(img_size), img_layer))
  else:
    img = np.vstack((img, img_layer))


#画像を表示する
cv2_imshow(img)    # google colab用
#画像を保存する
#cv2.imwrite(sav_path, img)    # save_pathに画像を保存する場所を書く

# routeの線がどのように繋がっているかの確認用
#import matplotlib.pyplot as plt
#plt.figure(figsize=(18, 9))
#plt.imshow(line_space, cmap="gray")
