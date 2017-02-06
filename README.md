# Vehicle Detection and Tracking

This repository contains the work I did for the vehicle detection and tracking project in the Udacity Self-Driving Car Nanodegree program. The objective of the project is to find cars in the images of a front looking camera, and track these cars. Goal is to find all cars as soon as they enter into view, while minimizing the chance of the occurence of 'false positives': Find a car where there isn't one. 

First a car finding algorithm has to be set up. This involves obtaining a collection of images of cars and non cars, and training a classifier that succesfully finds cars in images. With the classifier cars can be found in the image provided by the forward looking camera. As cars can be in different spots in an image and of different sizes, a sliding window technique with varying window sizes needs to be implemented. To correct for finding (too many) false positives and to build a smooth view on the surrounding cars, an averaging and thresholding method is used. In the project I made use of many Python functions provided in the Udacity SelfDriving Car NanoDegree lessons.

## Training data

Within the project a couple of data sources are available. I chose to combine all of them to have a broad collection of cars and notcars. I ended up with a set of 8792 images of cars and a set of 8968 images that contain images of roads and surroundings, without cars in them. Both sets are in RGB color space and of format 64x64 pixels. They are also of  approximatelythe same size, so no augmentation of one of the calsses is required.
Below are 5 examples of cars en 5 examples of non cars. 

<img src="https://cloud.githubusercontent.com/assets/23193240/22543697/0a763e38-e932-11e6-859e-6767d16b2a6f.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543698/0a8d830e-e932-11e6-9c46-fb242c1301d0.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543699/0a94c9de-e932-11e6-819e-be601985963d.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543700/0a959c74-e932-11e6-950f-0139c10eb307.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543789/5dc50182-e932-11e6-98e7-15e9a6b55b81.jpg" width="128" height="128" /> 

<img src="https://cloud.githubusercontent.com/assets/23193240/22543711/133bc4fc-e932-11e6-93e6-50b527f17432.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543713/133f3f2e-e932-11e6-8b7e-887529ea55e3.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543712/133db17c-e932-11e6-8737-c0a4c630509f.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543714/133f54aa-e932-11e6-8dd0-370e2d227a79.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543796/6819e67a-e932-11e6-8863-f57416bd8b7e.jpg" width="128" height="128" /> 

The data were labeled "1" for cars and "0" for non cars, and after creating a random sequence, split into 80% training data and 20% test data. The training data were fed to various collections of feature extractors - as described below - and afterward the feature values were normalized inorder to prevent one feature dominating the others.

## Training a classifier

For this project I chose to use a Support Vector Machine with a linear kernel. As there were many others parameters to tune I sticked with the standard parameters for the SVM, as they worked quite good from the beginning.
A couple of features are available to implement with this classifier: spatial binning of color, color histograms and histograms of oriented gradients (HOG). Each of these involves the selection of parameters. As especially the HOG feature has many parameters, I decided to investigate this one first with a simulation in which I varied a couple of parameters. In a following step I combined the three features and ran another simulation to arrive at parameters to use in the video pipeline.

### The HOG parameters

Preliminary research already proved the RGB wasn't very useful for using the HOG feature. HSV did a much better job, so I conducted the HOG research within this colorspace. Parameters varied for HOG were:
* layer within the colorspace (could also be all three together)
* the number of orientation bins
* number of pixels per cell
* number of cells per block
It became immediately clear that training an SVM on only HOG with only one layer of the colorspace provided much worse results than when using all layers. For brevity I left out the results on single layers. The results for the three layers combined is show in the table below: 


**# orientations**|**pix/cell**|**cells/block**|**HOG channel**|**test accuracy**
:-----:|:-----:|:-----:|:-----:|:-----:
13|16|2|ALL|0.9937
13|16|1|ALL|0.9900
**13**|**8**|**2**|**ALL**|**0.9986**
**13**|**8**|**1**|**ALL**|**0.9977**
13|4|2|ALL|0.9968
**13**|**4**|**1**|**ALL**|**0.9991**
9|16|2|ALL|0.9950
9|16|1|ALL|0.9887
9|8|2|ALL|0.9937
9|8|1|ALL|0.9955
**9**|**4**|**2**|**ALL**|**0.9977**
9|4|1|ALL|0.9959
5|16|2|ALL|0.9900
5|16|1|ALL|0.9855
5|8|2|ALL|0.9946
5|8|1|ALL|0.9932
5|4|2|ALL|0.9950
5|4|1|ALL|0.9941

In bold the highest accuracies. Trying to run the training on an even larger number of orientations wasn't succesful because of an unacceptable long training time. For the same reason I chose to go with 9 orientations (and 4 pixels per cell, 2 cells/block and ALL layers): The accuracy is not that much different from the highest accuracy features and, as processing time is important for creating a working video pipeline, this helps in reducing processor time.


I investigated In the next step (three feature optimalization) I will consider other color spaces, but for HOG I kept it to HSV, for complexity reasons. HOG was tested on various values of number of orientatations, pixels per cell, cells per block, and the color layer of the image. The results made immediately clear that the image layers need to be combined to arrive at a useful feature. Single layer accuracy never reached 0.99 whereas a combination of the three layers easily reached that figure. The table below shows the accuracy with varying parameters. I left out the results of the single channel HOG for brevity.  

### training the combinations of features


Some preliminary research made clear that HOG parameters are already providing good solutions. RGB is not a good color space, but the HSV, LUV, HLS, YUV and YCrCb all made sense to further investigate. Using a single layer of a color space degraded performance considerably, so will not be considered. Spatial binning made some difference both in accuracy as in training time. histogram increasing bins to 64 also increased accuracy and training time. Training time is a measure for performance using the feature in a video pipeline. So we want this to be small.

See what maximum accuracy we can achieve: imagesize 64x64 (source image resolution), 64 bins for histograms, try this on all color spaces mentioned above and on all combinations of feature vectors (spatial binning, color histograms, HOG). As the accuracy varied to some extent from training to training, I decided to run those combinations 10 times and average the accuracies, shown in the table below. 

training on a SVM with a linear kernel



**color space**|**spatial color binning**|**histograms of color**|**HOG**|**average test accuracy**|**standard deviation**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
HSV|FALSE|FALSE|TRUE|0.9800|0.0025
**HSV**|**FALSE**|**TRUE**|**FALSE**|**0.9972**|**0.0013**
HSV|FALSE|TRUE|TRUE|0.9961|0.0019
HSV|TRUE|FALSE|FALSE|0.9188|0.0052
HSV|TRUE|FALSE|TRUE|0.9908|0.0020
HSV|TRUE|TRUE|FALSE|0.9855|0.0020
**HSV**|**TRUE**|**TRUE**|**TRUE**|**0.9974**|**0.0012**
LUV|FALSE|FALSE|TRUE|0.9777|0.0038
LUV|FALSE|TRUE|FALSE|0.9911|0.0008
LUV|FALSE|TRUE|TRUE|0.9952|0.0013
LUV|TRUE|FALSE|FALSE|0.9410|0.0051
LUV|TRUE|FALSE|TRUE|0.9893|0.0024
LUV|TRUE|TRUE|FALSE|0.9718|0.0031
LUV|TRUE|TRUE|TRUE|0.9935|0.0017
HLS|FALSE|FALSE|TRUE|0.9775|0.0037
HLS|FALSE|TRUE|FALSE|0.9950|0.0011
HLS|FALSE|TRUE|TRUE|0.9966|0.0012
HLS|TRUE|FALSE|FALSE|0.9218|0.0059
HLS|TRUE|FALSE|TRUE|0.9902|0.0015
HLS|TRUE|TRUE|FALSE|0.9802|0.0028
HLS|TRUE|TRUE|TRUE|0.9954|0.0016
YUV|FALSE|FALSE|TRUE|0.9805|0.0030
YUV|FALSE|TRUE|FALSE|0.5454|0.0123
YUV|FALSE|TRUE|TRUE|0.9795|0.0024
YUV|TRUE|FALSE|FALSE|0.9376|0.0046
YUV|TRUE|FALSE|TRUE|0.9893|0.0020
YUV|TRUE|TRUE|FALSE|0.9394|0.0059
YUV|TRUE|TRUE|TRUE|0.9901|0.0017
YCrCb|FALSE|FALSE|TRUE|0.9778|0.0023
YCrCb|FALSE|TRUE|FALSE|0.5433|0.0083
YCrCb|FALSE|TRUE|TRUE|0.9792|0.0026
YCrCb|TRUE|FALSE|FALSE|0.9366|0.0045
YCrCb|TRUE|FALSE|TRUE|0.9901|0.0022
YCrCb|TRUE|TRUE|FALSE|0.9395|0.0035
YCrCb|TRUE|TRUE|TRUE|0.9904|0.0021

The highest accuracy is scored in the HSV color space, using all three features. Interesting is that using only the spatial color binning feature also provides a high accuracy. Both rows are shown in bold in the table above. The training of these classifiers was done with large parameters for image size and size of histogram bins. This will result in a relatively long processing time in the video pipeline, so it makes sense to find out how much the accuracy drops when those parameters are decreased. 
In a second round of classifier training only the two highest performing models from round 1 are considered. Image size is varied in 16x16, 32x32 and 64x64, while number of histogram bins is varied 32 and 64. The results, once again done on 10 training rounds per parameter set and averaged, are shown in the tables below.

**HSV color space, all three features combined**

**image size (pixels)**|**number of histogram bins**|**average test accuracy**|**standard deviation**
:-----:|:-----:|:-----:|:-----:
16x16|32|0.9961|0.0010
**16x16**|**64**|**0.9981**|**0.0006**
32x32|32|0.9971|0.0010
32x32|64|0.9977|0.0011
64x64|32|0.9950|0.0010
64x64|64|0.9961|0.0016

**HSV color space, only spatial color binning**

**image size (pixels)**|**number of histogram bins**|**average test accuracy**|**standard deviation**
:-----:|:-----:|:-----:|:-----:
16x16|32|0.9871|0.0021
16x16|64|0.9976|0.0008
32x32|32|0.9848|0.0019
**32x32**|**64**|**0.9978**|**0.0013**
64x64|32|0.9856|0.0019
64x64|64|0.9976|0.0012

What settings to choose? 16x16 images (with 64 histogram bins) somewhat surprisingly provide the highest accuracy of 0.998, when all three features are combined. This seems very high, but still means that 2 out of 1000 samples will be misclassified. Considered the number of frames that will be searched in a single image, and the number of images in a video stream, it is clear that many misclassifications will occur. While a classifier with mmultiple features and high resolutiuon will consume more processing time, leading to a longer conversion time of the video stream and ultimately to a processing time that is simply too long for use in real time vehicle detection. Also consider using spatial color bins only in the video pipeline. Proof of the pudding is in the eating: see what works best in the pipeline.

## Sliding window search

Cars appear smaller in the image the farther they are away. The range of the image to be searched for small instances of a car is relatively small as is the size of the car image. From a couple of pictures it appears that small cars fit in a box of 32x32 pixels, whereas nearby and therefore larger cars need up to 128x128 pixels. In general there are no cars to be expected in the range y < 400. Also far away cars tend to be distributed closely around the x-center of the images. I decided to aim my search for cars in the following way:

**type of car**|**size of box**|**y-range**|**x-range**|**step size**|**frames in y direction**|**frames in x direction**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
small|32x32|400 - 448|400 - 880|50%|2|29
medium|64x64|400 - 496|320 - 960|50%|2|19
large|128x128|400 - 592|192-1088|50%|2|13
 | | | | | | 
total number of frames| | | | |122| 







![found_boxes_false_pos](https://cloud.githubusercontent.com/assets/23193240/22651609/f36adc08-ec83-11e6-9c6a-b2d773c8811f.png)

![found_boxes_car_other_lane](https://cloud.githubusercontent.com/assets/23193240/22652640/7fcfd038-ec87-11e6-9b0d-e3fe639b8a43.png)



![heat_view](https://cloud.githubusercontent.com/assets/23193240/22664344/86928536-ecb0-11e6-8f29-822d956f1147.png)
![road_view](https://cloud.githubusercontent.com/assets/23193240/22664345/869338f0-ecb0-11e6-938c-46dd89b11989.png)

## sources
* udacity.com: various Python functions
