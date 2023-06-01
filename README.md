



<div align="center">
  <h1>Heart Disease Prediction</h1>
</div>


<div align="center">
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /></a>
    <a href="https://numpy.org"><img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" /></a>
    <a href="https://pandas.pydata.org"><img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" /></a>
    <a href="https://www.scipy.org"><img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white" /></a>
    <a href="https://matplotlib.org"><img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" /></a>
    <a href="https://seaborn.pydata.org"><img src="https://img.shields.io/badge/seaborn-%23565E64.svg?style=for-the-badge&logo=seaborn&logoColor=white" /></a>
    <a href="https://scikit-learn.org"><img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" /></a>
    <a href="https://www.tensorflow.org"><img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" /></a>
    <a href="https://keras.io"><img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white" /></a>
    <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-%235F4690.svg?style=for-the-badge&logo=streamlit&logoColor=white" /></a>
</div>
<br>

[Explore the Jupyter Notebook](https://github.com/nripstein/Heart-Disease-Prediction/blob/main/heart%20failure%20prediction%20notebook.ipynb)

[Try the Classification Website](https://heart-disease-prediction-ripstein.streamlit.app/)

# Table of Contents
- [Objective](https://github.com/nripstein/Heart-Disease-Prediction/blob/main/README.md#objective)
- [Usage](https://github.com/nripstein/Heart-Disease-Prediction/blob/main/README.md#usage)
- [Dataset](https://github.com/nripstein/Heart-Disease-Prediction/blob/main/README.md#dataset)
- [Exploratory Data Analysis](https://github.com/nripstein/Heart-Disease-Prediction/blob/main/README.md#exploratory-data-analysis)
- [Data Imputation Techniques for Null Value Replacement](https://github.com/nripstein/Heart-Disease-Prediction/blob/main/README.md#data-imputation-techniques-for-null-value-replacement)
- [Feature Selection With Inferential Statistics](https://github.com/nripstein/Heart-Disease-Prediction/blob/main/README.md#feature-selection-with-inferential-statistics)
- [Classification Models](https://github.com/nripstein/Heart-Disease-Prediction/blob/main/README.md#classification-models)

<details open>
  <summary><H2>Objective</H2></summary>
<p>WARNING: Please note that this project is intended as an illustrative example of the potential application of machine learning in assisting medical professionals with heart disease diagnosis. The information and results presented here (or on the accompanying website) do not constitute medical advice in any form.</p>
	
	
<p>Heart disease is a prevalent health condition that requires accurate and timely diagnosis for effective treatment. This project aims to develop a machine learning model to assist doctors in diagnosing heart disease accurately and efficiently. By leveraging statistical learning algorithms and a comprehensive dataset, the model can analyze various patient factors and provide predictions regarding the probability of heart disease. The implementation of diagnostic machine learning models like this one offers several potential benefits, including improved diagnostic accuracy, reduced burden on medical professionals, and early detection of disease. Furthermore, the project promotes data-driven medicine and contributes to ongoing efforts in machine learning-based medical diagnosis. By providing an additional tool for risk assessment and decision-making, I hope that models like this one can enhance patient outcomes and advance our understanding of heart disease.</p>
</details>


<details open>
  <summary><H2>Usage</H2></summary>
	<a href="https://nripstein-heart-disease-pre-heart-disease-prediction-app-wsdsib.streamlit.app/"><img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/45d41781-87cc-454d-bdab-2ddb09c539a1" alt="Animated GIF"></a>
Please note that the Heart Disease Prediction Website is intended for demonstration and educational purposes only and should not substitute professional medical advice or diagnosis.
	<ol>
	<li>A medical professional could collect and enter the necessary patient information (such as age, sex, chest pain type, resting blood pressure, etc.) into the website</li>
	<li>Once all the required data has been entered, click the "Predict" button to initiate the prediction process.</li>
	<li>A medical professional could interpret the prediction result alongside other diagnostic information and medical expertise to make informed decisions and provide appropriate care for the patient.</li>
		
</ol>
</details>


<details>
  <summary><H2>Dataset</H2></summary>

[Link to dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

**Attribute Information**
1.  Age: age of the patient [years]
2.  Sex: sex of the patient [M: Male, F: Female]
3.  ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4.  RestingBP: resting blood pressure [mm Hg]
5.  Cholesterol: serum cholesterol [mm/dl]
6.  FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
7.  RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
8.  MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
9.  ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
10.  Oldpeak: oldpeak = ST [Numeric value measured in depression]
11.  ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
12.  HeartDisease: output class [1: heart disease, 0: Normal]

**Source**

This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. The five datasets used for its curation are:

-   Cleveland: 303 observations
-   Hungarian: 294 observations
-   Switzerland: 123 observations
-   Long Beach VA: 200 observations
-   Stalog (Heart) Data Set: 270 observations

Total: 1190 observations  
Duplicated: 272 observations

`Final dataset: 918 observations`

Creators:
1.  Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2.  University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3.  University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4.  V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

Donor to UC Irvine Machine Learning Repository:  
David W. Aha (aha '@' ics.uci.edu) (714) 856-8779

The datasets from the above sources were combined into the dataset I used by Kaggle user fedesoriano.

Data source Citation:
  
fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [May 22, 2023] from [https://www.kaggle.com/fedesoriano/heart-failure-prediction](https://www.kaggle.com/fedesoriano/heart-failure-prediction).
</details>


<details open>
    <summary><h2>Exploratory Data Analysis</h2></summary>
    <p>Questions I Asked About the Data:</p>
    <ol>
        <details>
            <summary>How many positive and negative examples are there of the target variable?</summary>
	     <img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/697609a2-8ef0-4ea5-bf7a-a263102b9ed8" alt="target_freq" width="75%">
	     <p>The dataset is close to balanced, so there is no need to impliment techniques to improve classifaction of infrequent categories like Synthetic Minority Over-sampling.</p>
        </details>
        <details open>
            <summary>How are continuous variables distributed (in particular, are they normally distributed)?</summary>
	    <img src=https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/62b9a4ff-74a2-4a57-84e1-92a1a767425b" alt="continuous_distribution" width="75%">
	    <img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/745cbb3c-0248-4b97-baee-aa6b309bee99" alt="qq_plots" width="75%">
	      
<p><strong>Key Takeaways:</strong></p>  <ol>  <li>Upon visually examining the distribution of age, resting blood pressure, and maximum heart rate, they appeared to resemble a normal distribution. However, the application of Q-Q plots indicated deviations from Gaussian distribution. Consequently, I conducted Shapiro-Wilk tests on each of these variables, which confirmed their non-normal distribution.</li>  <li>Notably, a considerable number of cholesterol values were assigned as 0 to represent null values.</li>  </ol>  <p><strong>Leveraging These Insights:</strong></p>  <ol>  <li>To address the departure from normality, I opted to employ the <code>StandardScaler()</code> function from the sklearn library. This transformation aimed to bring the data points closer to a normal distribution.</li>  <li>Initially, when constructing the baseline models, I retained the original cholesterol data without any modifications. However, to overcome the limitation imposed by the null cholesterol values, I employed a series of techniques which aim to replace the null values with numbers from which models can generate meaningful predictions.</li>  </ol>
        </details>
        <details open>
            <summary>How do continuous variables change in conjunction with the target variable?</summary>
            <img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/eaf1a71b-2a5c-4345-824f-c433a36cadad" alt="continuous_target" width="75%">
	    <p>A visual inspection indicates that age, maximum heart rate and oldpeak are most different in Heart Disease positive vs negative.  <a href="https://github.com/nripstein/Heart-Disease-Prediction/#anova">This was later statistically confirmed with an ANOVA</a></p>
        </details>
        <details>
            <summary>How many examples are there of each categorical variable?</summary>
	    <img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/7e0ee7a7-1514-476c-983d-14ca90e77e42" alt="continuous_target" width="75%">
            <p>Answer goes here...</p>
        </details>
        <details>
            <summary>How does each categorical variable change in conjunction with the target variable?</summary>
            <br>
            <img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/d7aa282c-d841-4b64-806d-fb54b388b21f" alt="categorical_target" width="75%">
	    <p>Answer goes here...</p>
        </details>
    </ol>
</details>

<details open>
    <summary><h2>Data Imputation Techniques for Null Value Replacement</h2></summary>
    	<p>As I discovered during Exploratory Data Analysis, the dataset has 172 samples with null values for Cholesterol (which were initially set to 0). I explored various data imputation techniques in an attempt to extract meaningful training data from such samples.</p>
	<p>Initially, simple imputation strategies were deployed, namely: mean, median, and mode imputation. A noteworthy improvement in model performance was observed compared to models trained on the original dataset where null values were replaced by a default value of zero. Among these initial imputation techniques, mean imputation was found to deliver the best results for most machine learning models.</p>
	<p>Building upon these initial findings, I applied a sophisticated imputation method: applying regression analysis to estimate the missing values. The regression techniques applied included Linear Regression, Ridge Regression, Lasso Regression, Random Forest Regression, Support Vector Regression, and Regression using Deep Learning. Each of these regression-based imputation techniques displayed a similar level of performance in terms of RMSE and MAE.</p>
	<p>The performance of the regression models was found to be not as satisfactory as initially hypothesized, often falling short of the results obtained with mean imputation.  Despite this, it was observed that for Random Forest Classifiaction models, the regression-based methods exhibited strong results in terms of precision and specificity metrics. Particularly, Linear Regression and Ridge Regression-based imputation strategies performed well in these areas.</p>
</details>

<details open>
    <summary><h2>Feature Selection With Inferential Statistics</h2></summary>
	<p>I used inferential statistics to determine the importance of the dataset's features.  If I found that a feature has no significant impact on the target variable, then it would be helpful to try models which discard that variable.  Removing an insignificant vairbale would reduce noise in the data, ideally lowering model overfitting and improving classification accuracy. For continuous features, I conducted an ANOVA, and for categorical features, I used a Chi-Squared Test.</p>
	
<H3>ANOVA</H3>
Analysis of Variance (ANOVA) is a method from inferential statistics that aims to determine if there is a statistically significant difference between the means of two (or more) groups.  This makes it a strong candidate for determining importance of continuous features in predicting a categorical output.  I used a One-Way ANOVA to test the importance of each continuous feature by checking  whether presence of heart disease had a statistically significant effect on the feature's mean.  
<H4>ANOVA Results</H4>
I found that there was a statistically significant difference (p<0.05) for each continuous feature.  This led me to decide to keep all continuous features as part of my classification models.
<br>

<img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/b1fe7f56-d9b8-4149-a72a-3bf6f0b3bfda" alt="ANOVA_results" width="50%">
																	

   <h3>Chi-Squared Test</h3>
    
The Chi-Squared test is a statistical hypothesis test that is used to determine whether there is a significant association between two categorical variables. It compares the observed frequencies in each category of a contingency table with the frequencies that would be expected if the variables were independent. In the context of feature selection for my machine learning models, I used the Chi-Squared test  to identify the categorical features that are most significantly associated with the target variable.

   <H4>Chi-Squared Test Results</H4>
Like the continuous features, I found a statistically significant difference in heart disease (p<0.05) according to each categorical feature.  This led me to decide to keep all categorical features as part of my classification models.   
<img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/e82a5f1a-96b3-4f26-80d5-fe57dad0d480" alt="chi_sq_results" width="50%">
</details>


<details open>
    <summary><h2>Classification Models</h2></summary>
    <p>A key component of this project involved the implementation and performance evaluation of a variety of classification models. The chosen models were tested on the datasets prepared as described in the "Data Imputation Techniques for Null Value Replacement" section. All the datasets utilized in this phase had continuous features standardized and normalized to ensure a uniform scale and improved model performance. </p>
    <ol>
        <li>Logistic Regression</li>
        <li>Random Forest Classifier</li>
        <li>Support Vector Machine Classifier</li>
        <li>Gaussian Naive Bayes Classifier</li>
        <li>Bernoulli Naive Bayes Classifier</li>
        <li>XGBoost Classifier</li>
        <li>Neural Network (of various architectures)</li>
    </ol>
    <p>For the neural network models, an expansive exploration of hyperparameter variations was conducted using cross-validation. These architectures ranged from those with a single hidden layer to those with three hidden layers, with the number of neurons per layer varying from 32 to 128.</p>
    <p>Each of these models was trained on 80% of the data, and tested on 20%.  Accuracy, Precision, Recall, F1-Score and Specificity metrics were tracked.</p>
</details>


<details open>
  <summary><H2>Results and Model Performance Evaluation</H2></summary>
  <p>A thorough analysis of over 80 models was conducted in this project, with the evaluation criteria based on several metrics including accuracy, precision, recall (sensitivity), F1-score, and specificity. Out of all the models evaluated, two demonstrated superior performance in their respective contexts.</p>
  <ol>
  <li><b>Deep Learning Model:</b></li>
  <p>The top performing model by <b>Accuracy, Recall, </b>and<b> F1-Score</b> was a deep learning model trained on data where missing cholesterol values were imputed using the mean of the available values.  Performance metrics can be found in the below figures and in Tables 1 and 2.  To accompany the single-number metrics, PDFs were also constructed to quantify the uncertainty in this model's sensitivity/recall and specificity.  I wrote a comprehensive report detailing the generation of Sensitivity and Specificity PDFs and Credible Intervals which can be found in the repository, <a href="https://github.com/nripstein/Heart-Disease-Prediction/blob/main/Bayesian%20Approach%20to%20Assessing%20Uncertainty%20in%20Sensitivity%20and%20Specificity%20Metrics.pdf">or by clicking this link</a>
        <img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/801074f9-c7d8-4b4a-ab8e-ac19945553f1" alt="Deep Learning Classifier Confusion Matrix" width="65%">
        <img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/ff2014c7-9a2f-40be-a294-4183c662abbf" alt="Deep Learning Classifier Sensitivity and Specificity PDFs" width="65%">
      </p>


<p align="center">  <table>  <caption>Table 1: Deep Learning Model Performance</caption> <thead> <tr> <th>Accuracy</th> <th>Precision</th> <th>Recall</th> <th>F1-Score</th> <th>Specificity</th> </tr> </thead> <tbody> <tr> <td>91.30%</td> <td>91.96%</td> <td>93.64</td> <td>92.79</td> <td>87.84%</td> </tr> </tbody> </table> </p> 
  <table>
    <caption>Table 2: Deep Learning Model Sensitivity and Specificity CI</caption>
    <thead>
      <tr>
        <th></th>
        <th><strong>95% CI Minimum</strong></th>
        <th><strong>95% CI Maximum</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Sensitivity/Recall</strong></td>
        <td>88.5%</td>
        <td>96.5%</td>
      </tr>
      <tr>
        <td><strong>Specificity</strong></td>
        <td>80.5%</td>
        <td>93.5%</td>
      </tr>
    </tbody>
  </table>
</p>
		 <li> <b>Random Forest Classifier:</b></li>
<p>The top performing model by <b>Precision</b> and <b>Specificity</b> was a Random Forest Classifier trained on data with missing cholesterol values imputed using Ridge Regression. Performance metrics can be found in the below figures and in Tables 2 and 3.</p>
		 
<p align="center">
        <img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/17778c8c-7dd9-4997-8276-f03f90d2a3a1" alt="Random Forest Classifier Confusion Matrix" width="65%">
        <img src="https://github.com/nripstein/Heart-Disease-Prediction/assets/98430636/9fd600b4-2ee4-4bf0-8d95-9906ab359b85" alt="Random Forest Classifier Sensitivity and Specificity PDFs" width="65%">
      </p>


<p align="center">  <table>  <caption>Table 3: Random Forest Classifier Performance</caption> <thead> <tr> <th>Accuracy</th> <th>Precision</th> <th>Recall</th> <th>F1-Score</th> <th>Specificity</th> </tr> </thead> <tbody> <tr> <td>89.67%</td> <td>94.34%</td> <td>88.50</td> <td>91.32</td> <td>91.55%</td> </tr> </tbody> </table> 
  <table>
    <caption>Table 4: Random Forest Classifier Sensitivity and Specificity CI</caption>
    <thead>
      <tr>
        <th></th>
        <th><strong>95% CI Minimum</strong></th>
        <th><strong>95% CI Maximum</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Sensitivity/Recall</strong></td>
        <td>82.5%</td>
        <td>92.5%</td>
      </tr>
      <tr>
        <td><strong>Specificity</strong></td>
        <td>84.5%</td>
        <td>96.5%</td>
      </tr>
    </tbody>
  </table>
</p>
<p align="center">


</li>
  </ol>

</details>
