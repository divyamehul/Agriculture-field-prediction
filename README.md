# Agriculture-field-prediction

**Predicting quality of an agriculture field using Data Analysis and Machine Learning**

As a part of extended coursework in Machine Learning, we developed a model to predict quality of an agricultural field by analyzing a farmers’ dataset. This project was under the guidance of Prof. Vineeth N. Balasubramanian. The implementation steps were:

1. Conducted exploratory data analysis by analyzing data, visualizing data, and understanding the features

2. Conducted data pre-processing by handling missing data (imputations), normalizing the dataset, and one-hot encoding categorical variables

3. Conducted feature engineering based on correlation of features with target variable

4. Tried different models such as XGBoost, AdaBoost, Logistic Regression etc. Best performance was given by
a Random Forest.


**Details to use the model**

Unzip the train and test data from Data.zip file.

The code is written in the notebook attached. To run it, cd into the folder, and execute

"bash eval.sh es22btech11013"

OR

"python es22btech11013_foml24_hackathon.py --test-file ~/test.csv --predictions-file ~/output.csv"
