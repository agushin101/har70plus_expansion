# Predicting Activity in Elderly Populations
## Human activity recognition (HAR) expansion research for patients aged 70+

This project aims to expand the research done on the HAR70+ dataset (https://archive-beta.ics.uci.edu/dataset/780/har70) to create a more accurate model for human activity recognition on elderly subjects.

Human activity recognition (HAR) is the process of predicting the physical state of a person. Examples of such states include resting, walking, and ascending or descending stairs. A common technique to build a model to perform HAR is to attach sensors to a subject, ask that they perform a series of tasks, and use the sensors to record data about the subject’s movement as they execute their tasks.1 However, much of the data used to perform HAR is collected from young and middle-aged subjects. To examine the applicability of HAR to elderly populations, Ustad et al. performed a standard data-gathering procedure on a group of 18 individuals over the age of 70 to determine the efficacy of HAR on older populations.2 Specifically, they placed accelerometers on their subjects’ backs and thighs. From the data gathered by the accelerometers, they extracted acceleration in the x, y, and z directions from both the back and the thigh sensor. Their dataset consists of 18 .csv files, each one representing a different individual in their experiment. The first column in the files represents the timestamp at which a particular entry was recorded for an individual. The next three columns are the accelerations in the x, y, and z directions recorded by the sensor placed on the subject’s back. The three columns after that represent the same quantities recorded by the thigh sensor. The final column contains the label, an integer value that represents the physical activity the subject was engaged in at the time of the recording.

Ustad et al. limit their paper to testing the effects of an extreme gradient boost model
trained on a combination of their dataset with HARTH, a larger dataset that consists of
information gathered from individuals under the age of 70. We will contribute to this work by
assessing the efficacy of a variety of machine learning models on exclusively the dataset
generated by Ustad et al. to determine whether limiting the training data to only individuals
above the age of 70 improves testing accuracy on these individuals, something Ustad et al. did
not test. We will follow Stewart et al’s recommendations for preprocessing HAR data.3
Specifically, we will calculate several statistics such as the mean, standard deviation, and median
in both the time and frequency domain. We will use the Fast Fourier Transform to convert our
data into the frequency domain. Additionally, we will attempt to isolate the gravity and
movement components of our data by applying a Butterworth filter.
4 We will calculate these
statistics across every 5 second interval in the data, as was done by Ustad et al. We will use
principal component analysis to reduce the dimension of the feature space. After the
pre-processing, we will test several models, such as k-nearest neighbors, random forests, and support vector machines. We will also test the extreme gradient boost model to remain consistent
with Ustad et al’s procedure. We will assess the quality of a machine learning procedure by
examining its accuracy and F1 score. In doing so, we will determine whether a more precise
model can be created to perform HAR in elderly individuals.


Bibliography:
1. https://www.mdpi.com/1424-8220/21/23/7853
2. https://www.mdpi.com/1424-8220/23/5/2368
3. https://archive-beta.ics.uci.edu/dataset/780/har70 for the dataset we will be using
4. https://journals.lww.com/acsm-msse/Fulltext/2018/12000/A_Dual_Accelerometer_System_for_Classifying.25.aspx
5. See footnote 3.
