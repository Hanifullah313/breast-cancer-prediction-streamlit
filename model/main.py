import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle


def clean_data():
    data = pd.read_csv('C:\\Users\\Hanif ullah laptop\\Desktop\\Breast Cancer Prediction Project\\data\\data.csv')
    # DROP  UNNECESSARY COLUMNS
    data = data.drop(columns=['Unnamed: 32', 'id'])
    # diagnosis column encode
    data["diagnosis"] = data["diagnosis"].map({'M': 1, 'B': 0})
    return data


def create_model(data):

    # defining x and y
    x = data.drop(columns=['diagnosis'])
    y = data['diagnosis']

    # feature scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # training the model
    model = LogisticRegression()
    model.fit(x_train, y_train) 


    # test model accuracy
    y_pred = model.predict(x_test)
    print("Accuracy Score ",accuracy_score(y_test , y_pred))
    print("classification Metrics : \n",classification_report(y_test , y_pred))
    return model , scaler 

def main():
    data = clean_data()
    model , scaler = create_model(data)
    # save the model to disk
    with open('model\logistic_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    # save the scaler to disk
    with open('model\scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)


if __name__ == "__main__":
    main()