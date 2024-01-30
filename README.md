## Predictive modeling of pedestrians' visual attention in urban environment using 360° video

This is code and dataset of the paper "Entropy-Based Guidance and Predictive Modelling of Pedestrians’ Visual Attention in Urban Space"

### Visualization Demo
Visualization of 120°-front-view model (top-ground truth, bottom-prediction)
<div align="center">
<img src="demo/comparison.gif" width="500px" align="center"/>
</div>

<div align="center">
<img src="demo/comparison_02.gif" width="500px" align="center"/>
</div>

Visualization of 360° model (top-ground truth, bottom-prediction)
<div align="center">
<img src="demo/comparison360.gif" width="500px" align="center"/>
</div>

<div align="center">
<img src="demo/comparison360_02.gif" width="500px" align="center"/>
</div>


### Environment

python>=3.8.16

torch >=2.0.1

opencv-python>=4.7.0.72

### Dataset

* Dataset of 120°-front-view

* Dataset of 360°

### Train

To train 120°-front-view model, run: main_train.py

To train 360° model, run: mian_train_360.py
