# ResTR

이번 프로젝트에서는  **이미지를 보고 미적으로 좋은 이미지와 나쁜 이미지로 분류하는 딥러닝 모델**을 연구한다.

미적 품질 평가(Aesthetic quality assessment) 모델은 인공지능의 하위 분야 중 하나인 미적 컴퓨팅(Computational Aesthetics)에서 연구되는 주제다.

이 프로젝트에서는 인간이 미를 평가하는 방법을 크게 세 가지 로 가정하였다,
- `요소`
- `요소 간 관계`
- `요소의 배치`

기존 미적 평가 모델에서 사용하는 **CNN(Convolution Neural Network)** 만으로 이루어진 신경망 구조로는 `요소` 분석만을 충족시킬 뿐 `요소 간 관계`, `요소의 배치` 이 두 가지 방법을 구현할 수 없다. 그래서 다음의 두 가지 신경망을 더 추가하여 기존 CNN의 한계를 극복하였다.

- 새로운(Novel) 신경망 구조 `TN(Total versus dot neural network)`
- **Deepmind**팀의 [`RN(Relation network)`](https://arxiv.org/abs/1706.01427)



이번 프로젝트에서는 pre-trained 된 resnet50을 이용한 `ResTR(Resnet50 + TN + RN)`이라는 새로운 모델을 제안한다. 
> Resnet50 with TN(Total versus dot Network) and RN(Relation Network)

- `Resnet50` (요소 그 자체를 분석)
- `TN` (요소의 배치 : 전체 이미지와 요소 간 관계를 분석)
- `RN` (요소 간 관계를 분석)

ResTR은 이미지 내의 `요소들의 관계`에 대해서 중점적으로 분석하는 모델이다. 


## 성능
AVA 데이터셋을 기반으로 한 성능 측정에서 `ResTR(Resnet50 + TN + RN)`은 **타 논문의 최고 성능의 모델(ILGNet-Inc.V4)**에 비해 **AVA1**에서 `0.22%`, **AVA2**에서 `4.16%` 높은 성능을 기록하였다. 
> State of the art result at 2018/11/21 

아래는 이번 연구의 내용과 결과물을 간략히 요약한 포스터 이미지다.
> 물리학과라는 정체성을 살려 디자인하였다. 반도체에서 전자가 전자대에서 가전자대로 > 이동하는 모습을 디자인하였으며, 사용된 이미지는 AVA 데이터셋의 이미지들이다.

![posteriamge](https://github.com/IllgamhoDuck/ResTR/blob/master/aesthetic.jpg)

## Why did I started this?
> 모방을 넘어서서 창의적인 디자인이 가능한 인공지능은 만들어질 수 있을까?

이런 의문에서 시작된 인공지능 프로젝트다. 이는 충분히 가능하며 `디자인 생성 모델`과 `미적 평가 모델` 이 두 모델이 필요하다. 이 논문에서는 창의적 디자인을 하는 인공지능 연구의 선행되는 연구로서 두 가지 모델 중 `미적 평가 모델`을 먼저 만들고자 한다.

## AVA Dataset
> A Large-Scale Database for Aesthetic Visual Analysis

미적 평가 모델을 만드는데 사용되는 데이터 셋은 AVA(A Large Scale Database for Aesthetics visual Analysis)[1]다. AVA는 255,529장으로 이루어진 데이터 셋으로 사진 작가들의 콘테스트 사이트인 www.dpchallenge.com 에서 크롤링된 이미지로 이루어져 있다.

각 사진마다 1점부터 10점의 점수까지 사람들에게 투표 받은 횟수가 기록되어 있다. 각 이미지 당 총 투표 횟수는 78 ~ 549회 사이에 형성되어 있으며 평균은 210회이다. 이를 통해 각 이미지마다 평균 점수를 계산할 수 있으며, 5점 이상인 이미지는 좋은 이미지, 5점 미만의 이미지는 나쁜 이미지로 분류하였다. 이렇게 분류된 좋은 이미지는 180,856장이며, 나쁜 이미지는 74,673장이다.

다른 논문들과의 공정한 비교를 위해 AVA를 이용해 만든 AVA1, AVA2데이터셋을 사용했으며 https://github.com/BestiVictory/ILGnet 에서 사용된 데이터 셋을 그대로 가져왔다.

## Reference
> 1. Naila Murray, Luca Marchesotti, Florent Perronnin, “AVA: A large-scale
database for aesthetic visual analysis”, in IEEE, June 16-21, 2012
> 2. Adam Santoro , David Raposo , David G.T. Barrett, "A simple neural network module for relational reasoning", arXiv:1706.01427, 5 June 2017
> 3. Muktabh Mayank Srivastava, Sonaal Kant , "Visual aesthetic analysis using deep neural network: model and techniques to increase accuracy without transfer learning",
In arXiv, 9 Dec 2017
> 4. Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton, "Dynamic Routing Between Capsules", arXiv:1710.09829, 7 Nov 2017
> 5. Xin Jin, Le Wu, Xiaodong Li, Siyu Chen, Siwei Peng, Jingying Chi, , Shiming Ge, Chenggen Song, Geng Zhao, "Predicting Aesthetic Score Distribution through Cumulative Jensen-Shannon Divergence", arXiv:1708.07089, 20 Nov 2017
> 6. Xin Jin, Le Wu, Xiaodong Li, Xiaokun Zhang, Jingying Chi, Siwei Peng, Shiming Ge, Geng Zhao, Shuying Li, "ILGNet: Inception Modules with Connected Local and Global Features for Efficient Image Aesthetic Quality Classification using Domain Adaptation", arXiv:1610.02256, 29 Apr 2018
> 7. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun ,"Deep Residual Learning for Image Recognition", arXiv:1512.03385, 10 Dec 2015
> 8. Hossein Talebi, Peyman Milanfar, "NIMA: Neural Image Assessment", arXiv:1709.05424, 26 Apr 2018
> 9. Xin Lu, Zhe Lin, Hailin Jin, Jianchao Yang, James Z. Wang,"RAPID: Rating Pictorial Aesthetics using Deep Learning", 2014
> 10. R. Datta, D. Joshi, J. Li, J. Wang, "Studying aesthetics in photographic images using a computational approach", In European Conference on Computer Vision (ECCV), pages 288–301, 2006.
> 11. Y. Ke, X. Tang, F. Jing, "The design of high-level features for photo quality assessment", In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), volume 1, pages 419–426, 2006.
> 12. Y. Luo, X. Tang, "Photo and video quality evaluation: Focusing on the subject", In European Conference on Computer Vision (ECCV), pages 386–399, 2008.
> 13. W. Luo, X. Wang, X. Tang, "Content-based photo quality assessment", In IEEE International Conference on Computer Vision (ICCV), pages 2206–2213, 2011
> 14. S. Bhattacharya, R. Sukthankar, M. Shah, "A framework for photo-quality assessment and enhancement based on visual aesthetics", In ACM International Conference on Multimedia (MM), pages 271–280, 2010.
> 15. S. Dhar, V. Ordonez, T. Berg, "High level describable attributes for predicting aesthetics and interestingness", In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1657–1664, June 2011.
33 / 33
> 16. M. Nishiyama, T. Okabe, I. Sato, Y. Sato, "Aesthetic quality classification of photographs based on color harmony", In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 33–40, 2011.
> 17.  P. O’Donovan, A. Agarwala, A. Hertzmann, "Color compatibility from large datasets", ACM Transactions on Graphics (TOG), 30(4):63:1–12, 2011.
> 18. Ashish V erma, Kranthi Koukuntla, Rohit V arma, Snehasis Mukherjee, "Automatic Assessment of Artistic Quality of Photos", arXiv:1804.06124, 17 Apr 2018
> 19. Shuang Ma, Jing Liuy, Chang Wen Chen, "A-Lamp: Adaptive Layout-Aware Multi-Patch Deep Convolutional Neural Network for Photo Aesthetic Assessment", arXiv:1704.00248, 2 Apr 2017
> 20. Shu Kong, Xiaohui Shen, Zhe Lin, Radomir Mech, Charless Fowlkes, "Photo Aesthetics Ranking Network with Attributes and Content Adaptation", arXiv:1606.01621, 27 Jul 2016
> 21. Weining Wang, Mingquan Zhao, Li Wang, Jiexiong Huang, Chengjia Cai, and
Xiangmin Xu, “A multi-scene deep learning model for image aesthetic evaluation”,
Signal Processing: Image Communication, pp. –, 2016
> 22. Geoffrey Hinton, Sara Sabour, Nicholas Frosst, "MATRIX CAPSULES WITH EM ROUTING", In ICLR, 2018
> 23. Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio, "Generative Adversarial Nets", In NIPS, 2014
> 24. Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning", arXiv:1602.07261, 23 Aug 2016
> 25. The SciPy community, numpy.fft.fft2, 24 Jul 2018, https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/

## License
MIT License

Copyright (c) 2019 illgamho_duck

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.