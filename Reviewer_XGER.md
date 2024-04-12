# Reviewer XGER

## [Q3.1] Insight into the Change in Hyperparameters

### Explanation sensitivity to hyperparameter alpha 

### Average faithfulness comparison of various alpha values in Time and Time-Frequency Domain for all datasets. Meand and standard deviation are shown across three classifiers. The boldface highlights the best performance in each domain.

|  | Time | Time-Frequency |
|----------------------|----------|----------|
| &alpha; = 0.8             | 0.115 &plusmn; 0.06   | 0.122 &plusmn; 0.06   |
| &alpha; = 0.5             | 0.131 &plusmn; 0.06   |  0.133 &plusmn; 0.05  |
| &alpha; = 0.2             | **0.147 &plusmn; 0.04**   |  **0.154 &plusmn; 0.08**  |

The alpha value indicates the insertion weight, and 1-alpha denotes the deletion weight according to equation 11 in the paper. The values above indicate mean and standard deviation for all nine UCR datasets using all three classifiers. The results show that decrease in the insertion weight and increase in the deletion weight increases the average faithfulness values for both time and time-frequency domain. The average faithfulness value is obtained by calculating the class probability change when inserting the most important feature into a baseline RBP data (insertion) or deleting the most important feature from original data (deletion). We hypothesize the reason higher deletion weight leads to better performance is because in the deletion method, the original data is within the model’s training distribution, and deleting a key feature effectively moves the data out of the training distribution leading to significant probability change. However, in the insertion method, the baseline RBP data is already outside the model’s training distribution, and inserting a key feature does not lead to drastic probability change by moving into the training distribution.

---

### Adjusting the window size and hop size 

We measured the average Faithfulness@k across all three models (bi-LSTM, ResNet-34, Transformer) with adjusted window size and hop size to observe the effect on the performance results. From the original window size of 16 and hop size of 8, we conducted experiments in size increased multiple of four and eight, which are window size 64, hop size 32, and window size 128, hop size 64. We conduct these experiments on the CinCECGTorso dataset of the UCR repository, which contains time-series length of 1639 timesteps, the longest dataset in our experiments. The main objective of conducting this experiment on the longest dataset is to first show that our Combined method has the optimal performance for various window sizes and hop sizes, and second is to show that the overall scale of the faithfulness values go up due to the larger area covered by the time-frequency features. 

### Average Faithfulness@k values for various methods in CinCECGTorso dataset. Mean and standard deviation are shown across three classifiers. The boldface highlights the best performance in each @k. The window size is 64, and the hop size is 32.   

|  | @1 | @2 | @4 | @6 | @8 |
|----------------------|----------|----------|----------|----------|----------|
| LIME                | **0.064 &plusmn; 0.04**   | 0.098 &plusmn; 0.08   | 0.291 &plusmn; 0.09   | 0.577 &plusmn; 0.03   | 0.688 &plusmn; 0.05   |
| KernelSHAP                | 0.059 &plusmn; 0.06   | 0.104 &plusmn; 0.07  | 0.295 &plusmn; 0.08   | 0.541 &plusmn; 0.06   | 0.653 &plusmn; 0.07   |
| RISE                | 0.062 &plusmn; 0.08   | 0.092 &plusmn; 0.08  | 0.274 &plusmn; 0.09  | 0.562 &plusmn; 0.09  | 0.671 &plusmn; 0.07   |
| Insertion                | 0.055 &plusmn; 0.09   | 0.101 &plusmn; 0.06  | 0.291 &plusmn; 0.05  | 0.560 &plusmn; 0.06  | 0.647 &plusmn; 0.04   |
| Deletion                | 0.062 &plusmn; 0.06   | 0.105 &plusmn; 0.07  | **0.313 &plusmn; 0.06**  | 0.575 &plusmn; 0.05  | 0.683 &plusmn; 0.08   |
| Combined                | 0.060 &plusmn; 0.05   | **0.108 &plusmn; 0.05**  | 0.308 &plusmn; 0.08  | **0.583 &plusmn; 0.04** | **0.691 &plusmn; 0.06**   |

### Average Faithfulness@k values for various methods in CinCECGTorso dataset. Mean and standard deviation are shown across three classifiers. The boldface highlights the best performance in each @k. The window size is 128, and the hop size is 64.  

|     | @1              | @2              | @4              | @6              | @8              |
|------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| LIME       | 0.155 ± 0.02    | **0.340 ± 0.06**| 0.508 ± 0.03    | 0.681 ± 0.02    | 0.810 ± 0.07    |
| KernelSHAP | 0.150 ± 0.01    | 0.332 ± 0.04    | 0.502 ± 0.04    | 0.695 ± 0.04    | 0.834 ± 0.05    |
| RISE       | 0.151 ± 0.02    | 0.328 ± 0.06    | 0.496 ± 0.02    | 0.674 ± 0.04    | 0.818 ± 0.06    |
| Insertion  | 0.148 ± 0.01    | 0.325 ± 0.05    | 0.485 ± 0.03    | 0.659 ± 0.05    | 0.802 ± 0.04    |
| Deletion   | 0.153 ± 0.02    | 0.336 ± 0.06    | 0.514 ± 0.05    | 0.703 ± 0.07    | 0.842 ± 0.03    |
| Combined   | **0.159 ± 0.02**| 0.332 ± 0.03    | **0.524 ± 0.04**| **0.725 ± 0.04**| **0.855 ± 0.04**|

---