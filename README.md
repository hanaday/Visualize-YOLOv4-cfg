# Visualize-YOLOv4-cfg
YOLOv4のcfgファイルを分かりやすく視覚化する
(darknet:https://github.com/AlexeyAB/darknet)

YOLOv4のcfgファイルの中身を読み解きたかったので作成

- google colaboratory用に作ったが85行目と416行目のcv2_imshow辺りを変えれば他でも使えると思う

- 39～44行目のview_cfgはYOLOv4のcfgの構造をテキストの形で分かりやすく表示できないか試していたものなので消してもいいかもしれない

- 画像は縦と横が40×400の画像を層の数だけ積み重ねていく作り
- covolution層の長方形の内側に書かれている数字がフィルターの数で、外側に書かれているのが画像サイズの大きさ（画像サイズは縦と横が同じ大きさと想定）
- 黒色の線がroute、赤色の線がshortcut、青色の線がscale_channnels、緑色の線がsamの繋がりを表している

実際に生成された画像 (https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg より作成)


![yolov4_cfg](https://user-images.githubusercontent.com/79752527/148654137-137643e8-51e4-4abf-bc02-70b21eddb92b.png)
