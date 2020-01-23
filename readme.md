## DHS net

DHS net for salient objects detection(Pytorch implementation).
Some part of this project is based on
codes from https://github.com/wlguan/DHSNet-PyTorch ,
and this project is an optimized version.

### Requirements
Original running environment:
* Python 3.7.5
* Pytorch 1.3.1
* TorchVision 0.2.1
* pillow 7.0.0

See requirements.txt for detail.

### Training
1. Put corresponding dataset in ./input/
    * training images(RGB, jpg format): ./input/train/raw/
    * training masks(gray, png format): ./input/train/mask/
    * validation images(RGB, jpg format): ./input/test/raw/
    * validation masks(gray, png format): ./input/test/mask/
2. Run train.py, if you want to change some parameters,
see train.py for detail.

### Inference
1. Put inference data in ./inference/
    * inference images(RGB, jpg format): ./inference
2. Run inference.py, output saliency maps will be in
./output directory.
