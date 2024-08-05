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

# Dataset 
Download <span style="color: blue; text-decoration: underline;">Univariate Weka formatted ARFF files and .txt files</span> from [here](https://www.timeseriesclassification.com/dataset.php), unzip and place the directory in the data folder. The list of datasets used are as follows:
```
CincECGTorso, TwoPatterns, MixedShapes, Arrowhead, Strawberry, Yoga, Ford A, Ford B, GunpointMaleFemale
```

# Classifier
We provide code for ResNet-34, bi-LSTM, and Transformer model training shown in the paper. 

## Training
To train a classifier, run <code style="background-color: #E8E8E8;">trainer.py</code> with configurations below 
```
python trainer.py --mode train --dataset <dataset> --model_type <model_type> --num_classes <num_classes> --task <task>
```

## Testing
To return precision, recall, f1, and accuracy of trained classifier 
```
python trainer.py --mode test  --classification_model <classification_model> --dataset <dataset> --model_type <model_type> --num_classes <num_classes> --task <task>
```

* <code style="background-color: #E8E8E8;">dataset</code>: The name of the dataset (e.g. "arrowhead")
* <code style="background-color: #E8E8E8;">model_type</code>: The type of classifier model ("resnet", "transformer", "bilstm")
* <code style="background-color: #E8E8E8;">num_classes</code>: Integer value of the number of classes in the dataset 
* <code style="background-color: #E8E8E8;">task</code>: Classification or SpectralX ("classification", "spectralx")
* <code style="background-color: #E8E8E8;">classification_model</code>: Path to trained classification model




