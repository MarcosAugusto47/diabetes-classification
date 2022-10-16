import numpy as np
import pickle

with open('models/ensemble_model.pickle', 'rb') as f:
    loaded_pipe = pickle.load(f)

def predict_diabetes(new_data):
    # Predict diabetes
    predictions = loaded_pipe.predict(new_data)

    pred_to_label = {0: 'Negative', 1: 'Positive'}

    # Make a list of predictions.
    data = []
    for t, pred in zip(new_data, predictions):
        data.append({'pred': int(pred), 'label': pred_to_label[pred]})

    return data

if __name__=="__main__":
    # some samples
    new_data = np.array([[1, 109, 56, 21, 135, 25.2, 0.833, 23],
                         [1, 109, 56, 21, 135, 25.2, 0.833, 24]])

    predictions = predict_diabetes(new_data)
    print(predictions)