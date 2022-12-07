# NeuralNetwork-YOLOXNano-HandSeals-Recognition
Build Naruto Hand Seals Recognition with Convolutional Neural Network YOLOX-Nano

<!--# Introduction
-->
# Requirements
* onnxruntime 1.10.0 or Later
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later (Only when running SSD or EfficientDet)

# DataSet
1. <span id="cite_note-4">Kaggle Public dataset：[naruto-hand-sign-dataset](https://www.kaggle.com/vikranthkanumuru/naruto-hand-sign-dataset)</span>

# Trained Model
* YOLOX-Nano

# Directory
<pre>
│  simple_demo.py
│  
├─model
│  └─yolox
│      │ yolox_nano.onnx
│      └─yolox_onnx.py
│              
├─setting─┬─labels.csv
│         └─jutsu.csv
│      
├─utils
│          
└─_legacy
</pre>

# Usage
```bash
python simple_demo.py
python Ninjutsu_demo.py
```

In addition, the following options can be specified when running the demo.
<details>
<summary>Option specification</summary>
   
* --device<br>
Camera device number<br>
Default：
    * simple_demo.py：0
    * Ninjutsu_demo.py：0
* --file<br>
Video file name ※If specified, the video will be loaded in preference to the camera<br>
Default：
    * simple_demo.py：None
    * Ninjutsu_demo.py：None
* --fps<br>
Processing FPS ※Valid only if the inference time is less than FPS<br>
Default：
    * simple_demo.py：10
    * Ninjutsu_demo.py：10
* --width<br>
Width when shooting with a camera<br>
Default：
    * simple_demo.py：960
    * Ninjutsu_demo.py：960
* --height<br>
Height when shooting with a camera<br>
Default：
    * simple_demo.py：540
    * Ninjutsu_demo.py：540
* --skip_frame<br>
Whether to thin out when loading the camera or video<br>
Default：
    * simple_demo.py：0
    * Ninjutsu_demo.py：0
* --model<br>
Storage path of the model to load<br>
Default：
    * simple_demo.py：model/yolox/yolox_nano.onnx
    * Ninjutsu_demo.py：model/yolox/yolox_nano.onnx
* --input_shape<br>
Model input shape<br>
Default：
    * simple_demo.py：416,416
    * Ninjutsu_demo.py：416,416
* --score_th<br>
Class discrimination threshold<br>
Default：
    * simple_demo.py：0.7
    * Ninjutsu_demo.py：0.7
* --nms_th<br>
NMS threshold<br>
Default：
    * simple_demo.py：0.45
    * Ninjutsu_demo.py：0.45
* --nms_score_th<br>
NMS score threshold<br>
Default：
    * simple_demo.py：0.1
    * Ninjutsu_demo.py：0.1
* --sign_interval<br>
The hand-sign history is cleared when the specified time(seconds) has passed since the last mark was detected.<br>
Default：
    * Ninjutsu_demo.py：2.0
* --jutsu_display_time<br>
Time to display the Ninjutsu name when the hand-sign procedure is completed(seconds)<br>
Default：
    * Ninjutsu_demo.py：5
* --use_display_score<br>
Whether to display the hand-sign detection score<br>
Default：
    * Ninjutsu_demo.py：False
* --erase_bbox<br>
Whether to clear the bounding box overlay display<br>
Default：
    * Ninjutsu_demo.py：False
* --use_jutsu_lang_en<br>
Whether to use English notation for displaying the Ninjutsu name<br>
Default：
    * Ninjutsu_demo.py：False
* --chattering_check<br>
Continuous detection is regarded as hand-sign detection<br>
Default：
    * Ninjutsu_demo.py：1
* --use_fullscreen<br>
Whether to use full screen display(experimental function)<br>
Default：
    * Ninjutsu_demo.py：False
</details>

# License 
NeuralNetwork-YOLOXNano-HandSeals-Recognition is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

# License(Font)
KouzanMouhitsu(衡山毛筆) Font(https://opentype.jp/kouzanmouhitufont.htm)
