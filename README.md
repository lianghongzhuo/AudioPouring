# Making Sense of Audio Vibration for Liquid Height Estimation in Robotic Pouring
## Abstract
In this paper ([arXiv](https://arxiv.org/abs/1903.00650), [code](https://github.com/lianghongzhuo/AudioPouring), [video](https://www.youtube.com/embed/ZRUdyDkdspY)), we focus on the challenging perception problem in robotic pouring. Most of the existing approaches either leverage visual or haptic information. However, these techniques may suffer from poor generalization performances on opaque containers or concerning measuring precision. To tackle these drawbacks, we propose to make use of audio vibration sensing and design a deep neural network PouringNet to predict the liquid height from the audio fragment during the robotic pouring task. PouringNet is trained on our collected real-world pouring dataset with multimodal sensing data, which contains more than 3000 recordings of audio, force feedback, video and trajectory data of the human hand that performs the pouring task. Each record represents a complete pouring procedure. We conduct several evaluations on PouringNet with our dataset and robotic hardware. The results demonstrate that our PouringNet generalizes well across different liquid containers, positions of the audio receiver, initial liquid heights and types of liquid, and facilitates a more robust and accurate audio-based perception for robotic pouring.

## Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/ZRUdyDkdspY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

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
