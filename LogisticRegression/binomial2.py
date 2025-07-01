import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fix_the_file():
    df = pd.read_csv('MOCK_DATA-binomial2.csv')

    gender_map = {'Male': 0, 'Female': 1}
    bool_map = {True:1,False:0}

    df['gender'] = df['gender'].map(gender_map)
    df['heart_attack_last_5_years'] = df['heart_attack_last_5_years'].map(bool_map)

    df.to_csv('MOCK_DATA-binomial2-fixed.csv', index=False)


def binomial(test_size_input,random_state_input):
    df = pd.read_csv('MOCK_DATA-binomial2-fixed.csv')

    X = df.drop('heart_attack_last_5_years', axis=1)
    y = df['heart_attack_last_5_years']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_input, random_state=random_state_input)

    model = LogisticRegression(max_iter=10000, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    #print('Accuracy:', accuracy)
    return accuracy

#fix_the_file()

random_state_list = random.sample(range(1, 101), 5)
test_size_list = [0.01,0.1,0.2,0.5,0.75]

df_acc = pd.DataFrame()

for i in random_state_list:
    size_list = []
    for j in test_size_list:
        size_list.append(binomial(j,i))
    df_acc[i] = size_list

print(df_acc)