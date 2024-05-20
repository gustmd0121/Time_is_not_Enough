# Explanation Samples

Explanation Samples from UCR repository datasets for better illustration

---

## Arrowhead Dataset

![Alt text for your main image](Explanation_Samples/arrowhead/Avg.png)

**Average of test samples in each class**

The above figure represents the average of all test samples in each class of the Arrowhead dataset. This figure is provided as reference for the example explanation samples given below.   

---

## Class 1 Explanation Sample 69

<p float="left">
  <img src="Explanation_Samples/arrowhead/combined_69.png" />
  <img src="Explanation_Samples/arrowhead/deletion_69.png" />
  <img src="Explanation_Samples/arrowhead/insertion_69.png" />
  <img src="Explanation_Samples/arrowhead/lime_69.png" />
  <img src="Explanation_Samples/arrowhead/shap_69.png" />
  <img src="Explanation_Samples/arrowhead/rise_69.png" />
</p>

### Sample Explanation  

As denoted in the legend of each figure, **sample** represents the explanation sample and **masking** represents the inverse-STFT reconstructed signal after masking the most important feature from the time-frequency domain spectrogram. The most important feature is determined by each perturbation-based model stated in the title of each figure. The yellow-region represents the difference between the explanation sample and the masked reconstructed signal, which visualizes the effect of the important time-frequency feature.       

### Explanation Sample Comparison  

Although the average of each class does not indicate definite ground-truth important regions, a good class-specific explanation can be inferred by the distance between the class being explained to all other classes in the pinpointed region. Of the six different explanation figures above, the highlighted region of the combined method has the most distinct Class 1 region when referring to the average of each class. 

---

## Yoga Dataset

![Alt text for your main image](Explanation_Samples/yoga/Avg.png)

---

## Class 0 Explanation Sample 0

<p float="left">
  <img src="Explanation_Samples/yoga/combined_0.png" />
  <img src="Explanation_Samples/yoga/deletion_0.png" />
  <img src="Explanation_Samples/yoga/insertion_0.png" />
  <img src="Explanation_Samples/yoga/lime_0.png" />
  <img src="Explanation_Samples/yoga/shap_0.png" />
  <img src="Explanation_Samples/yoga/rise_0.png" />
</p>

### Explanation Sample Comparison  

The important region of the combined method has the most distinct Class 0 region when referring to the average of each class of Yoga dataset.


## Class 1 Explanation Sample 192

<p float="left">
  <img src="Explanation_Samples/yoga/combined_192.png" />
  <img src="Explanation_Samples/yoga/deletion_192.png" />
  <img src="Explanation_Samples/yoga/insertion_192.png" />
  <img src="Explanation_Samples/yoga/lime_192.png" />
  <img src="Explanation_Samples/yoga/shap_192.png" />
  <img src="Explanation_Samples/yoga/rise_192.png" />
</p>

### Explanation Sample Comparison  

The important region of the combined method has the most distinct Class 1 region when referring to the average of each class of Yoga dataset.

---

## Ford A Dataset

![Alt text for your main image](Explanation_Samples/forda/Avg.png)

---

## Class 1 Explanation Sample 302

<p float="left">
  <img src="Explanation_Samples/forda/combined_302.png" />
  <img src="Explanation_Samples/forda/deletion_302.png" />
  <img src="Explanation_Samples/forda/insertion_302.png" />
  <img src="Explanation_Samples/forda/lime_302.png" />
  <img src="Explanation_Samples/forda/shap_302.png" />
  <img src="Explanation_Samples/forda/rise_302.png" />
</p>

### Explanation Sample Comparison  

The Ford A dataset presents unique challenges in determining the optimal explanation through human perception because the class averages do not accurately show the distance between target class and other classes. Furthermore, the oscillation patterns in class 0 and class 1 are similar, leading to resemblances in their time-frequency characteristics. This similarity complicates the process of generating explanations based on time-frequency features. We provide explanation samples from each method above, but it is difficult to distinguish optimal important features with only human perception.   
