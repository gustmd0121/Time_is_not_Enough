# Reviewer Ves3 

## [b] Insufficiency of Human Evaluation

In order to strengthen our case of establishing the fairness of the ranking in explanation quality between our FIA combined method and other perturbation methods, we increased the number of evaluators from **six to twenty** graduate students studying machine learning. The results of this human evaluation indicates explanations produced by our method maintains superior ranking compared to other baseline methods despite increase in evaluators. The plot of human evaluation results is shown below:   

![Alt text for your main image](Human_Evaluation/Human_Eval.jpg)

Here, the rank-1 percentage increased for our Combined method from 42.9% to 46.6%, whereas LIME and SHAP rank-1 percentage tied in second place with 22.9%, and finally RISE rank-1 percentage was 21.6%. Compared to previous human evaluation with six graduate students, rank-1 percentage of LIME increased from 19.3% to 22.9%, RISE increased from 21.1% to 21.6%, and SHAP decreased from 25.4% to 22.9%.

The total user's evaluation scores are also calcuated by allocating 4 points for rank-1, 3 points for rank-2, 2 points for rank-3, and 1 point for rank-4. Note when conducting the user study we allowed equal rankings for two or more methods. The total user's evaluation scores are shown below: 

![Alt text for your main image](Human_Evaluation/Human_Eval2.jpg)

The total score of our combined method is 1172 points, the total score of SHAP is 997 points, the total score of RISE is 975 points, and finally the total score of LIME is 970 points. 

---

## Arrowhead Dataset

![Alt text for your main image](Explanation_Samples/arrowhead/Avg.png)

**Average of test samples in each class**

The above figure represents the average of all test samples in each class of the Arrowhead dataset. This figure is provided as reference for the example explanation samples given below.   

---
     

