# Robust Automatic Modulation Classification in Low Signal to Noise Ratio
# Introduction
This is the implementations related to our paper

T. T. An and B. M. Lee, "Robust Automatic Modulation Classification in Low Signal to Noise Ratio," in IEEE Access, vol. 11, pp. 7860-7872, 2023, doi: 10.1109/ACCESS.2023.3238995.

# Abstract

In this paper, we propose a threshold autoencoder denoiser convolutional neural network (TADCNN), which consists of a threshold autoencoder denoiser (TAD) and a convolutional neural network (CNN). TADs reduce noise power and clean input signals, which are then passed on to CNN for classification. The TAD network generally consists of three components: the batch normalization layer, the autoencoder, and the threshold denoise. The threshold denoise component uses an auto-learning threshold sub-network to compute thresholds automatically. According to experiments, AMC with TAD improved classification accuracy by 70% at low SNR compared with a model without a denoiser. Additionally, our model achieves an average accuracy of 66.64% on the RML2016.10A dataset, which is 6% to 18% higher than the current AMC model.

![image](https://user-images.githubusercontent.com/95015972/219922235-7c552a5c-6e75-4391-871e-31d9d5e99039.png)

![image](https://user-images.githubusercontent.com/95015972/219922243-07bb7b2b-cfde-4cfe-ad75-179b24f8e731.png)

# Result

![image](https://user-images.githubusercontent.com/95015972/219922264-96179e9b-7465-4ba1-844d-8b8689b827a9.png)

![image](https://user-images.githubusercontent.com/95015972/219922276-30bd771f-208d-4a11-867d-9d19834df3ef.png)


# Requirements and Installation

We recommended the following dependencies.

Python 3.9.12

TensorFlow 2.9

# Citation Format
If the implementation helps, you might citate the work with the following foramt:
@ARTICLE{10024264,
  author={An, To Truong and Lee, Byung Moo},
  journal={IEEE Access}, 
  title={Robust Automatic Modulation Classification in Low Signal to Noise Ratio}, 
  year={2023},
  volume={11},
  number={},
  pages={7860-7872},
  doi={10.1109/ACCESS.2023.3238995}}

