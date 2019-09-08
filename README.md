# CV_DL_Reading_List
A reading list related to our current project in SenseTime

* Section Links:
  * [Learning Iterative Optimization Solver for SLAM](#learning-iterative-optimization-solver-for-slam)
  * [Learning Iterative Network for SLAM](#learning-iterative-network-for-slam)
  * [RNN for Motion Estimation](#rnn-for-motion-estimation)
  * [CNN for Depth Estimation](#cnn-for-depth-estimation)
  * [Learning based Visual-Inertial Odometry](#learning-based-visual-inertial-odometry)
  * [Depth Estimation from Partial Observation](#depth-estimation-from-partial-observation)
  * [Visual SLAM without Learning](#visual-slam-without-learning)
  * [Adaptive Frame Selection from Video](#adaptive-frame-selection-from-video)
  * [Robotic Manipulation and Perception](#robotic-manipulation-and-perception)
  * [Other Interesting Papers](#other-interesting-papers)


## Learning Iterative Optimization Solver for SLAM
* Taking a Deeper Look at the Inverse Compositional Algorithm, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lv_Taking_a_Deeper_Look_at_the_Inverse_Compositional_Algorithm_CVPR_2019_paper.pdf), [code](https://github.com/lvzhaoyang/DeeperInverseCompositionalAlgorithm)
  * two-view feature encoder + convolutional M-estimator + trust region network, supervised, motion only

* BA-Net: Dense Bundle Adjustment Networks, *ICLR 2019*
  * [paper](https://openreview.net/pdf?id=B1gabhRcYX), [code](https://github.com/frobelbest/BANet)
  * learning the feature pyramid + the LM damping factor + the basis depth maps, supervised, motion and depth

* SceneCode: Monocular Dense Semantic Reconstruction using Learned Encoded Scene Representation, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhi_SceneCode_Monocular_Dense_Semantic_Reconstruction_Using_Learned_Encoded_Scene_Representations_CVPR_2019_paper.pdf)
  * compact code for optimization, supervised, motion and segmentation and depth

* CodeSLAM-Learning a Compact, Optimisable Representation for Dense Visual SLAM, *CVPR 2018*
  * [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Bloesch_CodeSLAM_--_Learning_CVPR_2018_paper.pdf)
  * compact code for optimization, supervised, motion and depth


## Learning Iterative Network for SLAM
* Learning to Solve Nonlinear Least Squares for Monocular Stereo, *ECCV 2018*
  * [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ronald_Clark_Neural_Nonlinear_least_ECCV_2018_paper.pdf)
  * LSTM-RNN for GN solver updates prediction (consider jacobian and residual terms), supervised, motion and depth

* DeepTAM: Deep Tracking and Mapping, *ECCV 2018*
  * [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Huizhong_Zhou_DeepTAM_Deep_Tracking_ECCV_2018_paper.pdf)
  * TODO
  
* DeMon: Depth and Motion Network for Learning Monocular Stereo, *CVPR 2017*
  * [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ummenhofer_DeMoN_Depth_and_CVPR_2017_paper.pdf)
  * bootstrap net + iterative net (CNN, DOES NOT consider jacobian and residual terms), supervised, motion and depth and flow


## RNN for Motion Estimation
* Beyond Tracking: Selecting Memory and Refining Poses for Deep Visual Odometry, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xue_Beyond_Tracking_Selecting_Memory_and_Refining_Poses_for_Deep_Visual_CVPR_2019_paper.pdf)
  * tracking + remembering + refining, memory augmented LSTM-RNN, supervised, motion only
  
* End-to-end, sequence-to-sequence probabilistic visual odometry through deep neural networks, *IJRR 2017*
  * [paper](https://journals.sagepub.com/doi/abs/10.1177/0278364917734298)
  * LSTM-RNN, supervised, motion only


## CNN for Depth Estimation
* Digging Into Self-Supervised Monocular Depth Estimation, *ICCV 2019*
  * [paper](https://arxiv.org/abs/1806.01260), [code](https://github.com/nianticlabs/monodepth2)
  * per-pixel minimum reprojection loss, auto-masking stationary pixels, full-resolution multi-scale, self-supervised

* Neural RGB->D Sensing: Depth and Uncertainty from a Video Camera, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Neural_RGBrD_Sensing_Depth_and_Uncertainty_From_a_Video_Camera_CVPR_2019_paper.pdf), [code](https://github.com/NVlabs/neuralrgbd)
  * depth probability distribution + bayesian filters + adaptive damping + kalman filter, DPV fusion, supervised, motion and depth
  
* Competitive Collaboration: Joint Unsupervised Learning of Depth, Camera Motion, Optical Flow and Motion Segmentation, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ranjan_Competitive_Collaboration_Joint_Unsupervised_Learning_of_Depth_Camera_Motion_Optical_CVPR_2019_paper.pdf), [code](https://github.com/anuragranj/cc)
  * TODO

* Depth from Videos in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras, *arXiv 2019*
  * [paper](https://arxiv.org/abs/1904.04998)
  * unsupervised, motion and depth and camera intrinsics and occlusion

* Learning Depth from Monocular Videos using Direct Methods, *CVPR 2018*
  * [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_Depth_From_CVPR_2018_paper.pdf), [code](https://github.com/MightyChaos/LKVOLearner)
  * differentiable DVO, unsupervised, motion and depth
  
* GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose, *CVPR 2018*
  * [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yin_GeoNet_Unsupervised_Learning_CVPR_2018_paper.pdf), [code](https://github.com/yzcjtr/GeoNet)
  * unsupervised, motion and depth and flow
  
* Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints, *CVPR 2018*
  * [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Mahjourian_Unsupervised_Learning_of_CVPR_2018_paper.pdf), [code](https://github.com/tensorflow/models/tree/master/research/vid2depth)
  * unsupervised, 2D photometric + 3D ICP losses, motion and depth
  
* Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction, *CVPR 2018*
  * [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhan_Unsupervised_Learning_of_CVPR_2018_paper.pdf), [code](https://github.com/Huangying-Zhan/Depth-VO-Feat)
  * unsupervised, stereo training, deep feature reconstruction loss, motion and depth (without scale ambiguity)

* Deep Virtual Stereo Odometry: Leveraging Deep Depth Prediction for Monocular Direct Sparse Odometry, *ECCV 2018*
  * [paper](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Nan_Yang_Deep_Virtual_Stereo_ECCV_2018_paper.pdf)
  * TODO

* UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning, *ICRA 2018*
  * [paper](https://ieeexplore.ieee.org/abstract/document/8461251)
  * unsupervised, stereo training, scale recovery, motion and depth (without scale ambiguity)

* Unsupervised Learning of Depth and Ego-Motion from Video, *CVPR 2017*
  * [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf), [code](https://github.com/tinghuiz/SfMLearner)
  * unsupervised, monocular training, 2D photometric loss, motion and depth
  
* Unsupervised Monocular Depth Estimation With Left-Right Consistency, *CVPR 2017*
  * [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf), [code](https://github.com/mrharicot/monodepth)
  * unsupervised, stereo training, appearance matching & left-right disparity consistency loss, depth only


## Learning based Visual-Inertial Odometry
* Selective Sensor Fusion for Neural Visual-Inertial Odometry, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.pdf)
  * visual-inertial fusion, LSTM-RNN, supervised, motion only

* Unsupervised Deep Visual-Inertial Odometry with Online Error Correction for RGB-D Imagery, *TPAMI 2019*
  * [paper](https://ieeexplore.ieee.org/document/8691513/)
  * iterative CNN, consider camera-imu synchronization errors, unsupervised
  
* VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem, *AAAI 2017*
  * [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14462/14272)
  * IMU-LSTM + Core-LSTM, consider camera-IMU calibration & synchronization Errors, supervised


## Depth Estimation from Partial Observation
* Estimating Depth from RGB and Sparse Sensing, *ECCV 2018*
  * [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhao_Chen_Estimating_Depth_from_ECCV_2018_paper.pdf), [code](https://github.com/kvmanohar22/sparse_depth_sensing)
  * TODO

* Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image, *ICRA 2018*
  * [paper](https://arxiv.org/pdf/1709.07492.pdf), [code](https://github.com/fangchangma/sparse-to-dense)
  * TODO

* Sparse Geometry from a Line: Monocular Depth Estimation with Partial Laser Observation, *ICRA 2017*
  * [paper](https://ieeexplore.ieee.org/document/7989590)


## Visual SLAM without Learning
* BAD SLAM: Bundle Adjusted Direct RGB-D SLAM, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Schops_BAD_SLAM_Bundle_Adjusted_Direct_RGB-D_SLAM_CVPR_2019_paper.pdf)
  * TODO
  
* ICE-BA: Incremental, Consistent and Efficient Bundle Adjustment for Visual-Inertial SLAM, *CVPR 2018*
  * [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_ICE-BA_Incremental_Consistent_CVPR_2018_paper.pdf), [code](https://github.com/baidu/ICE-BA)
  * TODO
  
* Hybrid Camera Pose Estimation, *CVPR 2018*
  * [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Camposeco_Hybrid_Camera_Pose_CVPR_2018_paper.pdf)
  * TODO
  
* Visual SLAM: Why Bundle Adjust?, *ICRA 2019*
  * [paper](https://arxiv.org/abs/1902.03747)
  * rotation averaging + known rotation BA
  
* (Add ORB_SLAM, DSO, VINS-Mono Here)


## Adaptive Frame Selection from Video
* Multi-Agent Reinforcement Learning Based Frame Sampling for Effective Untrimmed Video Recognition, *ICCV 2019*
  * [paper](https://arxiv.org/abs/1907.13369)
  * TODO

* BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Griffin_BubbleNets_Learning_to_Select_the_Guidance_Frame_in_Video_Object_CVPR_2019_paper.pdf), [code](https://github.com/griffbr/BubbleNets)
  * relative performance prediction + bubble sorting, video object segmentation using best annotation frame

* AdaFrame: Adaptive Frame Selection for Fast Video Recognition, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.pdf)
  * memory-augmented LSTM (selection, reward prediction, utility), video recognition using less frames

* Efficient Video Classification Using Fewer Frames, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Bhardwaj_Efficient_Video_Classification_Using_Fewer_Frames_CVPR_2019_paper.pdf)
  * TODO

* Watching a Small Portion could be as Good as Watching All: Towards Efficient Video Classification, *IJCAI 2018*
  * [paper](https://www.ijcai.org/proceedings/2018/0098.pdf), [code](https://github.com/hehefan/video-classification)
  * TODO

## Robotic Manipulation and Perception
* 6-DOF GraspNet: Variational Grasp Generation for Object Manipulation, *ICCV 2019*
  * [paper](https://arxiv.org/abs/1905.10520)
  * TODO

* U4D: Unsupervised 4D Dynamic Scene Understanding, *ICCV 2019*
  * [paper](https://arxiv.org/abs/1907.09905)
  * TODO

* Deep Hough Voting for 3D Object Detection in Point Clouds, *ICCV 2019*
  * [paper](https://arxiv.org/abs/1904.09664)
  * TODO

* Sim-To-Real via Sim-To-Sim: Data-Efficient Robotic Grasping via Randomized-To-Canonical Adaptation Networks, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/James_Sim-To-Real_via_Sim-To-Sim_Data-Efficient_Robotic_Grasping_via_Randomized-To-Canonical_Adaptation_Networks_CVPR_2019_paper.pdf)
  * TODO

* CRAVES: Controlling Robotic Arm With a Vision-Based Economic System, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zuo_CRAVES_Controlling_Robotic_Arm_With_a_Vision-Based_Economic_System_CVPR_2019_paper.pdf)
  * TODO

* FlowNet3D: Learning Scene Flow in 3D Point Clouds, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_FlowNet3D_Learning_Scene_Flow_in_3D_Point_Clouds_CVPR_2019_paper.pdf)
  * TODO

* A Robust Local Spectral Descriptor for Matching Non-Rigid Shapes With Incompatible Shape Structures, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_A_Robust_Local_Spectral_Descriptor_for_Matching_Non-Rigid_Shapes_With_CVPR_2019_paper.pdf)
  * TODO
  
## Other Interesting Papers
* DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds, *CVPR 2019*
  * [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_DeepMapping_Unsupervised_Map_Estimation_From_Multiple_Point_Clouds_CVPR_2019_paper.pdf), [code](https://ai4ce.github.io/DeepMapping/)
  * TODO

* Understanding the Limitations of CNN-based Absolute Camera Pose Regression, *CVPR 2019*
  * [paper](https://arxiv.org/abs/1903.07504), [datasets](https://github.com/tsattler/understanding_apr)
  * TODO

* Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving, *CVPR 2019*
  * [paper](https://arxiv.org/abs/1812.07179), [code](https://github.com/mileyan/pseudo_lidar)
  * pseudo-LiDAR representations + mimick LiDAR signal
