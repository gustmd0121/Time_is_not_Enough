# Time is Not Enough: Time-Frequency based Explanation for Time-Series Black-Box Models

![Model Architecture](figures/overall_figure.png)

Despite the massive attention given to time-series explanations due to their extensive applications, a notable limitation in existing approaches is their primary reliance on the time-domain. This overlooks the inherent characteristic of time-series data containing both time and frequency features. In this work, we present Spectral eXplanation (SpectralX), an XAI framework that provides time-frequency explanations for time-series black-box classifiers. This easily adaptable framework enables users to “plug-in” various perturbation-based XAI methods for any pre-trained time-series classification models to assess their impact on the explanation quality without having to modify the framework architecture. Additionally, we introduce Feature Importance Approximations (FIA), a new perturbation-based XAI method. These methods consist of feature insertion, deletion, and combination techniques to enhance computational efficiency and class-specific explanations in timeseries classification tasks. We conduct extensive experiments in the generated synthetic dataset and various UCR Time-Series datasets to first compare the explanation performance of FIA and other existing perturbation-based XAI methods in both time-domain and time-frequency domain, and then show the superiority of our FIA in the time-frequency domain with the SpectralX framework. Finally, we conduct a user study to confirm the practicality of our FIA in SpectralX framework for class-specific time-frequency based time-series explanations.

# Environment
Tested on
* PyTorch version >= 2.2.1
* Python version >= 3.10 
* NVIDIA GPU (CUDA 12.1)
* Clone the repository and install requirements: 
```
git clone https://github.com/gustmd0121/Time_is_not_Enough.git
cd Time_is_not_Enough
pip install -r requirements.txt
```
