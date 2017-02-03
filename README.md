# Vehicle Detection and Tracking





## Training a model for recognizing cars

used KITTI cars 5966 RGB images of size 64x64
non-vehicles 5068 RGB images of size 64x64

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

Some preliminary research made clear that HOG parameters are already providing good solutions. RGB is not a good color space, but the HSV, LUV, HLS, YUV and YCrCb all made sense to further investigate. Using a single layer of a color space degraded performance considerably, so will not be considered. Spatial binning made some difference both in accuracy as in training time. histogram increasing bins to 64 also increased accuracy and training time. Training time is a measure for performance using the feature in a video pipeline. So we want this to be small.

See what maximum accuracy we can achieve: imagesize 64x64 (source image resolution), 64 bins for histograms, try this on all color spaces mentioned above and on all combinations of feature vectors (spatial binning, color histograms, HOG). As the accuracy varied to some extent from training to training, I decided to run those combinations 10 times and average the accuracies, shown in the table below. 

training on a SVM with a linear kernel

**color space**|**spatial color binning**|**histograms of color**|**HOG**|**average test accuracy**|**standard deviation**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
HSV|FALSE|FALSE|TRUE|0.9800|0.0025
**HSV|FALSE|TRUE|FALSE|0.9972|0.0013**
HSV|FALSE|TRUE|TRUE|0.9961|0.0019
HSV|TRUE|FALSE|FALSE|0.9188|0.0052
HSV|TRUE|FALSE|TRUE|0.9908|0.0020
HSV|TRUE|TRUE|FALSE|0.9855|0.0020
**HSV|TRUE|TRUE|TRUE|0.9974|0.0012**
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
In a second round of classifier training only the two highest performing models from round 1 are considered. Image size is varied in 16x16, 32x32 and 64x64, while number of histogram bins is varied 32 and 64. The results, once again done on 10 training rounds per parameter set and averaged, is shown in the table below.


TABELLLLLLL

What settings to choose? Accuracy of 0.997 seems very high, but still means that 3 out of 1000 samples will be misclassified. Considered the number of frames that will be searched in a single image, and the number of images in a video stream, it is clear that many misclassifications will occur. On the other a classifier with high resolutiuon will consume more processing time, leading to a longer conversion time of the video stream and ultimately to a processing time that is simply too long for use in real time vehicle detection.
