# diabetes-classification
Machine Learning application that predicts if a woman, given some covariates values, has diabetes or not. The dataset used and some description can be found at https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database.

The application returns 0 (Negative) or 1 (Positive) via a POST request, that needs to be structured like this:

{  
    "Pregnancies": 0,  
    "Glucose": 137,  
    "BloodPressure": 40,  
    "SkinThickness": 35,  
    "Insulin": 168,  
    "BMI": 43.1,  
    "DiabetesPedigreeFunction": 2.288,  
    "Age": 33  
    }
  
More samples can be set via the POST request.

The models were trained, using Logistic Regression, with some variations and are all written on the jupyter notebook.

The application uses Docker and can be runned via the command docker run -p5000:5000 marcosaugusto47/diabetes-app