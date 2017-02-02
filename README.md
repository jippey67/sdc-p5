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

See what maximum accuracy we can achieve: imagesize 64x64 (source image resolution), 64 bins for histograms, try this on all color spaces mentioned above and on all combinations of feature vectors (spatial binning, color histograms, HOG)

|   |   |   |   |   |
|---|---|---|---|---|
|   |   |   |   |   |
|   |   |   |   |   |
|   |   |   |   |   |

