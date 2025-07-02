# Logistic Regression
Logistic regression is a machine learning algorithm used for classification - that is, determing whether an input belongs to a pre-defined class. It does this by first being trained on large datasets featuring several variables, and using these to determine the probability that the new input will fall into the chosen category.
## Description
This repo will be split into a number of sections, as I go over multiple stages and types of LR. This will be much more explaining how it works rather than changing any code, as the method is mostly set, though there are things we can do to play around with it.

# Getting the data
The first step is retrieving our dataset. In my case, I used a website called mockaroo -
```https://mockaroo.com/```

Here, you can generate a dataset of up to 1000 entries based on fields you enter, and can even have an AI agent create them based on a description, which is handy as larger datasets will improve our overall accuracy, though there is an issue which I will explain later on.

# Simple Regression
The first file, 'binomial.py', is a very basic use case for binomial LR - that is, the chosen category needs to be one of two options. Does a coin land on heads or tails? Does it rain or does it not rain? Does a car turn left or right? Any question that can be answered with one of two options is a use case for binomial LR.

In this instance, the scenario is for students having an exam. There are several variables, including attendance, study hours and mock exam scores, but the specific category in this case is the final exam score. In this case, the pass grade is set to 70. For our purposes, this also requires an additional column with a value of 0 or 1, denoting a pass or a fail, which can be set up before you download the dataset.

To train our model, we read in the dataset, then split it into two parts - everything used to predict the outcome (X) and the actual outcome (Y). This is then split further into train and test components, based on a given test size, as well as a 'random state' integer, which helps to randomise how the two are split. By default, this is None, but you can input one yourself if you want it to be consistent.

From there, the model is trained on the 'train' parts of the dataset, then predicts an output based on the 'test' component - specifically, the X_test component, giving us y_pred, or our prediction of what Y should be based on X. Finally, this new Y is compared to y_test, our actual output from X_test. These are then compared, giving us our final measure - accuracy. 
Accuracy is the measure of how well our model compares to the actual test data, up to %100, or 1. In this instance, with this dataset, our accuracy is 0.99, suggesting 99% accuracy. This sounds impressive, but remember, whether or not someone has passed, the Y component, is partly based on the final exam score, which is part of the X component. We could drop the pass column, and select the final score as our category, but this is no longer a binomial question, as the score can vary wildly. Actually doing this drops our score to 0.016, or %1.6. Point being, one cannot simply throw any dataset into the algortihm - additional work needs to be done, an example of which will be in the next section.

# Further steps
In this section, we will add some additional steps, both to improve the readability of our dataset, and the accuracy of our readout - that is, not specifically the accuracy of the model, but our reading of what the actual accuracy is.

For this case, we will be predicting the likelihood that a person, based on age, weight, habits and exercise routines, will suffer a heart attack, using data from patients, some of which have suffered heart attacks previously. The file for this is binomial2.py.

Our first issue is the data we have recieved - we want to drop the id column, as it doesn't actually help with our investigation, and the 'sex' and 'Suffered' columns are in types that we can't currently use, strings and booleans. To resolve this, we can drop the id column entirely, in the same way we drop the target column for testing. As for the other two, we can set up a function to clean the dataset -

```def fix_the_file():
    df = pd.read_csv('MOCK_DATA-binomial2.csv')

    sex_map = {'Male': 0, 'Female': 1}
    bool_map = {True:1,False:0}

    df['sex'] = df['sex'].map(sex_map)
    df['heart_attack_last_5_years'] = df['heart_attack_last_5_years'].map(bool_map)

    df.to_csv('MOCK_DATA-binomial2-fixed.csv', index=False)
```
    
Now, the sexes and booleans are numbers. In the case that your categories are more varied, you can map them to more numbers - only the target category needs to be a binary.

That done, setup is much the same as before - we split the dataset, train it, then compare the prediction to the actual output. Here, I've added a function to run through multiple test sizes and states, to show how changing these can affect the accuracy of the model -
```random_state_list = random.sample(range(1, 101), 5)
test_size_list = [0.01,0.1,0.2,0.5,0.75]

df_acc = pd.DataFrame()

for i in random_state_list:
    size_list = []
    for j in test_size_list:
        size_list.append(binomial(j,i))
    df_acc[i] = size_list

print(df_acc)
```
In this case, 5 random numbers are used to shuffle the dataset before splitting them. Then, for each number, the test size is changed, the test sample being larger each time, resulting in the follwing output -

```         7         50        49        1         32
0  0.600000  0.400000  0.800000  0.700000  0.500000
1  0.550000  0.470000  0.580000  0.530000  0.580000
2  0.545000  0.485000  0.560000  0.500000  0.535000
3  0.508000  0.512000  0.516000  0.518000  0.476000
4  0.526667  0.509333  0.522667  0.505333  0.505333
```
In this case, our accuracy is around %50, with the top row varying so wildy due to the small test data size of %1, suggesting that our model will predict correctly around half the time. This is likely due to my test data being randomly generated, rather than curated from actual patient data, as any real-world correlation would be much more apparent.

# Multinomial

Another form of LR we can carry out is Multinomial regression, in 'multinomial.py'. This is used in cases where your categories are in 2 or more unordered classes - for instance, different animals, different cars, different weather types, etc. Setup is much the same, though in this case I will be using a dataset featured with scikit-learn, as it avoids the potential issues outlined with my other datasets earlier. I have also featured the Kneighbours classifier, as another metric to check the accuracy of our model

```
Accuracy: 0.9685185185185186
KNN score: 0.9907407407407407
```
Here, we can see that the accuracy is very high in both cases, meaning that our model is likely very accurate, helped in no small part by the dataset used in this example.

# Exporting Models

That's all well and good, but as is, all we're doing is training models. How do we go about actually using them in a real world example?

For this we need an additional model - pickle, which is included in Python. Going through export.py, we can set up our model just as before, but now export it in a 'pkl' file. Then, in import.py, we recollect it, then apply it to our test data, giving an accuracy of
```
Accuracy: 0.9685185185185186
```

The same score as in our multinomial example.

To be extra sure that this is not a fluke, we can actually change the distribution of test data by changing the random_state variable as well, changing which data is put into the test or train components.

Setting it to 25 gives an accuracy of -
```
Accuracy: 0.9907407407407407
```

And note, in import.py at no point do we retrain the model. We only make use of a reshuffled testing dataset, and still have a very high accuracy. So here we've just created, exported, imported and tested a new LR model, which can make accurate predictions from our dataset.

# Final thoughts

This is a relatively straightforward example of using Machine Learning tools in your workflow, and would typically be done with much larger, more complex datasets, but the underlying logic is much the same. These tools can then be incorprated into much larger pipelines, or developed further with visualisations or multi-tiered operations.