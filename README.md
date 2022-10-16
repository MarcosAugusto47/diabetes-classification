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

The final model is an ensemble method that uses Logistic Regression, K-Nearest Neighbors and Random Forest, that uses a voting rule classifier between those three models. The Logistic Regression was tuned finding the best regularization technique (Lasso or Ridge) and the respective hyperparameter. Also, it was applied standard scaling for this specific model.

The application uses Docker and can be runned via the command (the image is published in my Docker Hub account):  
docker run -p5000:5000 marcosaugusto47/diabetes-app