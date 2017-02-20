#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/class_distribution.png "Visualization"
[image4]: ./examples/ahead_only.png "Traffic Sign 1"
[image5]: ./examples/speed_limit_60.png "Traffic Sign 2"
[image6]: ./examples/stop.png "Traffic Sign 3"
[image7]: ./examples/dangerous_curve_right.png "Traffic Sign 4"
[image8]: ./examples/end_of_no_passing.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 2nd code cell of the IPython notebook.  

I used the vanilla python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 3rd code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of classes in the training and testing sets:

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 4th code cell of the IPython notebook.

I converted the image to grayscale initially and played around with it a little bit, however the accuracy performance is not as ideal as I would expect. I then tried removing it from the pipeline and it seems the colored images are more easy to predict! Therefore I removed this step from the pipeline. I think the logic behind this is color is actually an important feature of traffic signs and containing useful information, sometimes the message they try to convey are blended into their colors.

The prepocessing technique I used is normalization. I applied what mentioned in the lecture and subtracted the RGB channels by 128, and then divided by 128, which normalizes the values to between (-1, 1) for numerical stability

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data split is already done in the newer version of the data. However if we were to split the training data, we could use sklearn's split utility to do this

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images. This is reflected in the 5th cell of the ipython notebook.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 6th cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride,  outputs 10x10x16      			|
| RELU          		|        										|
| Max pooling 			| 2x2 stride,  outputs 5x5x16        			|
| Flatten       		| outputs 400        							|
| Fully connected		| outputs 120        							|
| RELU          		|        										|
| Fully connected		| outputs 84        							|
| RELU          		|        										|
| Fully connected		| outputs 43        							|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 7th/8th/9th/10th cell of the ipython notebook. 

To train the model, I used the LeNet architecture by Yann LeCun.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 92.7%
* validation set accuracy of 92.7% 
* test set accuracy of 91.6%

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I chose the LeNet architecture learned through our lecture. I believe it would be suitable for this application for the following reasons:
* It uses convolutional layers which is essential for image feature extractions, this also made it easier to adjust the input/output shapes
* It has a decent amount of activation functions as well as several fully connected layers, this would make modeling nonlinear behaviours pretty easily.
* It returns the logits which is easy to feed into other parts of the pipeline
The final result of 91.7% accuracy on the test set without much tuning is actually quite surprising to me! I used a batch size around 80 and it performed pretty decent without too much overfitting within the 10 epochs, which is pretty impressive.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

There is a common shortcoming in these pictures: the contrast is too sharp. The white areas are pure zeros while the blues and reds are pure numbers as well. There is no noises in these images. They might be of good qualities (which is a good thing if we were not classifying images!), but because the training data have much lower contrasts and the brightness is also dim, I would assume these five images are hard to classify since the model has never seen 'pure' data like this before. The accuracy will be discussed in the next section.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| Priority Road   								| 
| Speed limit 60km/h    | General caution 								|
| STOP  				| Yield											|
| Dangerous curve right	| Priority Road					 				|
| End of no passing		| Priority Road      							|


The model was able to correctly guess 0 of the 5 traffic signs, which gives an accuracy of 0. However on the testing set earlier, the accuracy was 91.6%, thus it seems the model is overfitting and we need to train fewer epochs.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a priority road (probability of 0.99)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority Road   								| 
| 0     				| Keep right 									|
| 0 					| Turn left ahead								|
| 0 	      			| Yield     					 				|
| 0 				    | STOP              							|

Second:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .84         			| General Caution   							|
| .10     				| Go straight or left 							|
| .0 					| Traffic signals								|
| .0 	      			| Children crossing					 			|
| .0 				    | Beware of Ice/Snow     						|

Third:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .70         			| Yield     									| 
| .07     				| Roundabout mandatory 							|
| .07					| Double curve									|
| .05	      			| Speed limit 100km/h					 		|
| .03				    | Go straight or left      						|

Fourth:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority Road   								| 
| 0     				| Right-of-way next intersection 				|
| 0 					| Traffic signals								|
| 0 	      			| Children crossing					 			|
| 0 				    | Turn left ahead      							|

Fifth:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .40         			| Priority Road   								| 
| .27     				| Keep right 									|
| .20					| Right-of-way at the next intersection			|
| .04	      			| End of no passing					 			|
| .02				    | End of all speed and passing limits      		|


It is a little bit weird that the test accuracy is over 90% yet it cannot predict the new images right. My guess is that all the new images have white background which is different from the given training data. However in some of the cases the model seems to be pretty certain about its predictions. I am a little bit confused about this one and will continue to investigate this issue after the submission and continue to train the model. Thanks for taking your time reviewing this long assignment!