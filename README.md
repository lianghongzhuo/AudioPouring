## Abstract
In this paper ([arXiv](https://arxiv.org/abs/1903.00650), [code](https://github.com/lianghongzhuo/AudioPouring), [video](https://www.youtube.com/embed/Za8dDjGFE1k)), we focus on the challenging perception problem in robotic pouring. Most of the existing approaches either leverage visual or haptic information. However, these techniques may suffer from poor generalization performances on opaque containers or concerning measuring precision. To tackle these drawbacks, we propose to make use of audio vibration sensing and design a deep neural network PouringNet to predict the liquid height from the audio fragment during the robotic pouring task. PouringNet is trained on our collected real-world pouring dataset with multimodal sensing data, which contains more than 3000 recordings of audio, force feedback, video and trajectory data of the human hand that performs the pouring task. Each record represents a complete pouring procedure. We conduct several evaluations on PouringNet with our dataset and robotic hardware. The results demonstrate that our PouringNet generalizes well across different liquid containers, positions of the audio receiver, initial liquid heights and types of liquid, and facilitates a more robust and accurate audio-based perception for robotic pouring.

## Video
<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/Za8dDjGFE1k" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

## Code
Code of this project can be found at [https://github.com/lianghongzhuo/AudioPouring](https://github.com/lianghongzhuo/AudioPouring).


## Dataset
<p align="center">
<img src="images/setup.svg" width="65%" alt="setup" style="margin-left:auto;margin-right:auto;display:block">
</p>

- Contain video, audio, force/torque and position information collected during human pouring.
- Current, we offer dataset with only audio input.
- Download: [https://drive.google.com/open?id=1zavcGC73OTsV8bsYrk6kPArHhIJNvvF8]( https://drive.google.com/open?id=1zavcGC73OTsV8bsYrk6kPArHhIJNvvF8)

## Citation
If you found this paper useful in your research, please consider citing:

```plain
@article{liang2019AudioPouring,
  title={Making Sense of Audio Vibration for Liquid Height Estimation in Robotic Pouring},
  author={Liang, Hongzhuo and Li, Shuang and Ma, Xiaojian  and Hendrich Norman and Gerkmann Timo and Zhang, Jianwei},
  journal={arXiv preprint arXiv:1903.00650},
  year={2019}
}
```
