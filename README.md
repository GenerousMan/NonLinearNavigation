# NonLinear Interactive Video For Shopping Navigation
Code of paper 《Automatic Generation of Interactive Nonlinear Video for Online Apparel Shopping Navigation》.

<p align='center'>
    <img src="imgs/pipeline.png", width="800">
</p>

**Notably, using interactive video to implement shopping navigation is a new research topic, and our solution is of course not perfect. Please refer to our failure cases and limitations before using this repo.**

## Update
- [2022-10-25] Update README, to introduce out work.
- [2022-10-19] Create the project, and upload main codes of this paper.

## Abstract

We present an automatic generation pipeline of interactive nonlinear video for online apparel shopping navigation. Our approach is inspired by Google’s "Messy Middle" theory, which suggests that people mentally have two tasks of exploration and evaluation before purchasing. Given a set of apparel product presentation videos, our navigation UI will organize these videos for users’ product exploration and automatically generate interactive videos for user product evaluation. To support automatic methods, we propose a video clustering similarity (CSIM) and a camera movement similarity (MSIM), as well as a comparative video generation algorithm for product recommendation, presentation, and comparison. To evaluate our pipeline’s effectiveness, we conducted several user studies. The results show that our pipeline can help users complete the consumption process more efficiently, making it easier for users to understand and choose the product.


## Theoretical support —— Messy Middle
<p align='center'>
    <img src="imgs/messy_middle.jpeg", width="500">
</p>
Prior work has studied the behavioral logic of consumers and a "messy middle" theory was proposed, which noted that consumers often wander in the two states of exploration and evaluation when shopping online. Consumers explore their options and expand their consideration sets; then – either sequentially or simultaneously – they evaluate the options and narrow down their choices. Existing online shopping methods need to constantly switch pages to view and compare products. Such a shopping method reduces the exploration and evaluation efficiency and increases the time for customers to make a decision.

To shorten the shopping time between product exploration and decision-making, we propose an automatic approach for generating nonlinear videos into two-level, coarse-level exploration and fine-level evaluation, in support of online clothing shopping navigation. Our approach can automatically generate interactive nonlinear videos for product presentation and comparison based on consumers’ interactions.


## Requirements

``` bash
pip install requirements.txt
```

## Weights
- [AlphaPose](https://drive.google.com/drive/folders/1Fi_jvlc3kZUwwi6d6xikgd9ZGePTpYEh?usp=sharing)
- [Attribute and Category](https://drive.google.com/drive/folders/1vfO2GXi0wsJQ6zvPRSKOIMYLLAAvncuG?usp=sharing)
- [Unet](https://drive.google.com/drive/folders/1lssxarbbnPggT94pEnwqnVPSGpNdFAo1?usp=sharing)
- [ML Models](https://drive.google.com/drive/folders/1SZ3kZFp6NJGVMQkpYnc1Y_YcEjuRHqQu?usp=sharing)
- [Detail Classification](https://drive.google.com/drive/folders/1QgcrP0ZMKDTb8A2tHxonM_3mys-jkMVO?usp=sharing)
- [Landmark](https://drive.google.com/drive/folders/1BI7GLUjwdPguVpxneGPQK8dnPXkM_sBi?usp=sharing)
- [MaskROI](https://drive.google.com/drive/folders/1WPAnizJJ_tfr1Q5uESxzy3a_L0HPdK-B?usp=sharing)


## Usage - Video Association Algorithm

In the video association algorithm, we sample the input product video at intervals of $t$ frames and extract the feature $F=\{F^{cate}, F^{attr}, F^{color}\}$. We construct a graph through the average of all features $F$ and use this graph to show the associations of categories, attributes, and colors between clothing products. Video association algorithms can be divided into feature extraction and graph construction.

<p align='center'>
    <img src="imgs/exploration.jpg", width="800">
</p>

To recommend similar products (in many product videos), you can use this command:

``` bash
python examples/example_recommend.py
```

It may takes long time to extract features and build the products' graph. After calculating all features of all videos, we will save them in a .pickle file. If there are lots of video nodes, it will also take a lot of space to save the graph.


## Usage - Shot Association Algorithm

In the shot association algorithm, we automatically attach detail shots to the full shot. When consumers evaluate the product, they can click the area of interest in the video to obtain more targeted information. The algorithm can be divided into video shot classification, detailed shot classification, and keypoint detection.

<p align='center'>
    <img src="imgs/evaluation.png", width="800">
</p>


To associate close-up shots and full-shot, you can use this command:

``` bash
python examples/example_single_presentation.py
```

This example will only generate the keypoints' position in the full-shot, and the classification results of close-up. These results will be saved as a .yaml file. Through our player, we can play this .yaml file as an interactive video.




