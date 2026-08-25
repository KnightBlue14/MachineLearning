# Classifier

A classifier refers to a Machine Learning model that can make use of computer vision to 'see' inputs, be they pictures or video from static or live sources. In this folder, I will be going over how they can be used to 'read' playing cards and track faces.

## Description

A classifier uses similar principles to Logistic Regression, which I covered in another folder. By repeatedly exposing the model to an input, then testing it against another, similar input, it can be refined over multiple iterations to become very good, but not perfect, at predicting the output.

Similarly, a model can be used to track faces, even under different lighting conditions, different angles, different accessories, etc. In this case, I will be using a prebuilt model available with the dlib library, as it is general purpose enough to work with just a single image, rather than compiling thousands of them to build my own model.

# Playing Cards

In cards.ipynb, I first need to download an imageset, in this case from kagglehub. You can download from a browser, but you need an account with them. Doing it in the code block will dwonload it directly, though do note it will download to an automatically selected folder. You will need to move the files once downloaded.

After importing everything, we first build a class. There are over 7000 files to be used in building the model, so we will need a framework to quickly prepare them iteratively. That done, we can build a dictionary to assign the classes (the individual categories that our model is looking for - in this case, each type of playing card) to a number, in order to confirm that they are all readable. If there are 53 classes, including the Joker, then our dataset is ready to be used.

First, we need to build the basic skeleton of our model. After setting up another class for the model itself, we build it, specifying the number of classes, and setting up a simple loss function. The model will use this during training to improve it's fit, adjusting itself based on how wrong it's earlier guesses are. We can then begin training on our dataset, using the class established in the beginning to split the total dataset into 3 parts for training, testing and validation, then applying a dataloader to those parts to feed them into the model.

Then, we can begin the training loop. I have used 10 iterations for this, though you can use more or less as needed. Make sure the device is set to use your gpu, if available, or it will take longer. In the first phase, the model is exposed to the training dataset, and performance is judged based on the training loss. If it guesses correctly, this loss is smaller, but if it guesses incorrectly, the loss will be higher. Here we are using the Cross-Entropy Loss function to calculate the confidence of the prediction, then feeding it into the Adam optimiser to adjust the models learning process. Different functions will have their strengths and weaknesses, but these are good general use models for this purpose.

In the validation phase, we do the same thing, but using the validation dataset. The difference is that this is presenting cards that the model has not seen yet, as opposed to those it was trained on, checking it's ability to judge new material. This is important because it allows us to check for underfitting, in which case both losses would remain high and the model is performing poorly overall, but also overfitting, where training loss would be low but the validation loss high. This would occur if the model has been overtrained, meaning it is very good at checking the training data, but performs poorly in real world scenarios with data it has not been trained on.

Both losses are also added to lists to track them over time. In the graph, we can see that initially loss is very high, but drops over time, showing that the model is more accurately identifying cards as the number of iterations increases, bottoming out at around 7 across multiple loops of training. It isn't perfect, as this method means the confidence will never be 100%, but it can be trusted to guess correctly most of the time.

As a final showcase, we can pull up several random cards and have our model guess, allowing manual review of the overall accuracy. These are also from the test dataset, so they are completley foreign to the model. For the most part, it is able to correctly guess the card, even with ones that have a very distinct style or colour filter, though there are cards where it will struggle and suggest multiple options, mainly cards where the number or face symbol has been moved or altered in some way.

# Face tracking

Classifiers can also be used to identify and track specific individuals. In this case, I will be using it to track and identify the cast of the latest Mission Impossible film, using images from multiple red carpet events.

Starting with Tom Cruise, 'tom.png' is a standard picture from one of these events. In faces.ipynb, we can set up a function that will identify the face, marking it with keypoints, then draw a bounding box around it, simply a rectangle to visualise where the face is. We can then build a detector using pre-built modules and run this function on the picture via our detector, saving the result as 'output_tom.png'. This demonstrates the theory behind a basic classifier for this purpose, as well as allowing granular control over the box, including colour and thickness.

We can also use a similar method to identify crowds of people, an example being 'output_crowd', though for this repo I have removed that code, since what comes next is a more practical example.

During these events, the cast had several pictures taken, including one group selfie. Using a similar pipeline, we can not only draw boxes for all of them, but also identify them by name, by using other pictures to precompile a match.

First, we need to import our images for each cast member. The article I found for the selfie included most of the actors, though I have left one as 'Unknown' to showcase a contingency in case there is no match available. Then, we encode the images and add them to a list, as well as adding another list for each of their names.

Importing the group image, we can then perform another match, just as before. However, this will not match the faces to the names. To do this, we add another line to iterativley assign each actor's name to the face, comparing it to the encoding we did earlier to match them.

Then, we mark the boundaries for our bounding box, and add the names. By default, a face will be marked as 'Unknown', followed by a number. However, if a match is found, this will be overwritten by the actor's name. The box is then drawn, with the name on top, though this can be adjusted if needed. Finally, the output is saved as 'output_carpet.jpg', with all names correctly allocated with one unknown, even using images with different lighting conditions, camera angles and expressions.