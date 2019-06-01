# CV_DL_Reading_List
A reading list related to our current project in SenseTime

### 1. Learning Depth
* Neural RGB->D Sensing: Depth and Uncertainty from a Video Camera, *CVPR 2019*
  * [paper](https://arxiv.org/abs/1901.02571), [code](https://github.com/NVlabs/neuralrgbd)
  * depth probability distribution + bayesian filters + adaptive damping + kalman filter , supervised

* Learning Depth from Monocular Videos using Direct Methods, *CVPR 2018*
  * [paper](https://www.ci2cv.net/media/papers/Wang_Learning_Depth_From_CVPR_2018_paper.pdf), [code](https://github.com/MightyChaos/LKVOLearner)

* Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image, *ICRA 2018*
  * [paper](https://arxiv.org/pdf/1709.07492.pdf), [code](https://github.com/fangchangma/sparse-to-dense)

* Sparse Geometry from a Line: Monocular Depth Estimation with Partial Laser Observation, *ICRA 2018*
  * [paper](https://ieeexplore.ieee.org/document/7989590), [code]()
  
### 2. Learning Correspondence

### 3. Stereo System

### 4. Visual SLAM/Odometry
* Beyond Tracking: Selecting Memory and Refining Poses for Deep Visual Odometry, *CVPR 2019*
  * [paper](https://arxiv.org/abs/1904.01892), [code]()
  * tracking + remembering + refining, RNN, supervised

* BA-Net: Dense Bundle Adjustment Networks, *ICLR 2019*
  * [paper](https://openreview.net/pdf?id=B1gabhRcYX), [code](https://github.com/frobelbest/BANet)
  * learning the feature pyramid + the LM damping factor + the basis depth maps, supervised

* Taking a Deeper Look at the Inverse Compositional Algorithm, *CVPR 2019*
  * [paper](http://www.cvlibs.net/publications/Lv2019CVPR.pdf), [code](https://github.com/lvzhaoyang/DeeperInverseCompositionalAlgorithm)
  * two-view feature encoder + convolutional M-estimator + trust region network, supervised
  
* Visual SLAM: Why Bundle Adjust?, *ICRA 2019*
  * [paper](https://arxiv.org/abs/1902.03747), [code]()
  
### 5. Visual-Inertial Odometry
  
### 6. Others
* Understanding the Limitations of CNN-based Absolute Camera Pose Regression, *CVPR 2019*
  * [paper](https://arxiv.org/abs/1903.07504), [datasets](https://github.com/tsattler/understanding_apr)

* Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving, *CVPR 2019*
  * [paper](https://arxiv.org/abs/1812.07179), [code](https://github.com/mileyan/pseudo_lidar)
  * pseudo-LiDAR representations + mimick LiDAR signal
