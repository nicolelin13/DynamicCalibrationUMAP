# Dynamic of Calibration in Recommender System [UMAP'24]

Code for UMAP24 PAPER -- Beyond Static Calibration: The Impact of User Preference Dynamics on Calibrated Recommendation. We include data and code in folders as descripbed below. We also list detailed methodology description. 

## Directory
### Data
The pre-processed KuaiRec and GoodReads datasets are in the "Data" folder, each in the subfolder named after the dataset. Each dataset folder includes a series of datasets of sub-profiles within designed time windows. 

### Models 
All the recommendations are built upon well-tuned Bayesian Personalized Ranking (BPR) and ItemKNN. The tuning and modeling used in this work is based on the implementation of RecBole (https://github.com/RUCAIBox/RecBole). The tuning of key parameters is based on an exhaustive search for the best NDCG@10. The model statistics, including the best selected tuned parameter and accuracy-based measurement (recall@10 and NDCG@10), are available in the "Model" folder. 

### Analysis
Two jupyter notebooks are provided in the folder "Analysis", one for KuaiRec and one for GoodReads. Meanwhile, all the informative visualizations created during the analysis are in the subfolder "Figures." 


## Methodology Details
Our experiment workflow is as the follow diagram shows. 

### 1 Data prepparation
#### 1.1 KuaiRec
KuaiRec dataset is mostly the original fully observed dataset. In the original dataset, KuaiRec small matrix includes 4,676,570 interactions between 1,411 users and 3,327 videos. 

##### a. Preprocess 
Two basic preprocesses are applied: 1/ remove interactions without valid timestamps, 2/ create the binary preference signal attribute based on watch_ratio. After the full preprocess on KuaiRec, the full dataset is with 1,799,403 interactions between 1,411 users and 3,320 videos, and the data density is 0.62. 

Given the key research objective is to investigate the temporal calibration pattern, we exclude 181,992 interactions that are without valid timestamp values. 

The preference signal is watch_ratio, which is the play_duration divided by video_duration. The watch_ratio ranges from 0 to 571.52, where the mode is 0.77 and the mean is 0.91. As common sense applies, higher watch_ratio may indicate users like the given video. Referring to the mean of watch_ratio, we turned the signal into a binary by setting threshold value 0.9, where watch_ratio not lower than 0.9 is defined 1 and otherwise 0. 

##### b. Sub dataset creation 
Based on the preprocessed full dataset, 63 sub datasets were created by accumulating days, starting from the first 1 day to all 63 days. Refer to the appendix for basic dataset statistics for the sub datasets. The process is illustrated as the above figure. 

#### 1.2 GoodReads

### 2. Simulation Process
We conducted a simulation-based analysis for identifying the most representative segments of users' profiles that, if used for training the recommendation model, would yield more calibrated recommendation results. 

### 3. Other Key Experiment Basic Setup

To investigate the calibration pattern across different training time windows, we generate recommendations based on the training datasets in given time periods. Each recommendation is created via a specifically well-tuned model with the specific sub dataset. We applied two algorithms, BPR and ItemKNN, with tuning and training details listed below. Except for the miscalibration measurement, most of the experiments are conducted using the open-source recommendation library RecBole. 

#### 3.1 Model tuning 
With NDCG@10 as the objective, the hyperparameter tuning is via exhaustive search of given parameter ranges as provided below. Models for different time periods are independently tuned with the specific training set from the given time period. 
BPR:
Learning rate: [0.01, 0.05, 0.001, 0.005, 0.0001]
Embedding size: [64, 96, 128]
ItemKNN:
K (neighborhood size): [50, 100, 200]
Shrink (normalization hyper parameter in cosine distance calculation): [0, 0.5]

#### 3.2 Training and evaluation
For model training and evaluation with the well-tuned parameters, the dataset in a tested time period is randomly split into train, validation and test sets by the ratio 8:1:1. We focus on two metrics in this experiment, NDCG@10 (ranking accuracy) and miscalibration. NDCG@10 value is based on the value returned from RecBole, while miscalibration is calculated via the distribution comparison between the recommendation list and the preference shown in the given time period. 

Also, to reduce the random noise from experiments, we changed random seeds and repeated 6 times of the training and evaluation procedure mentioned above. The final evaluation results presented in the paper are based on the mean value from the repeated experiments.   

#### 4. User Segmentation
Given that users' varied behavioral patterns in the same dataset may be related to different degrees of calibration across individuals, we extend our analysis of calibration dynamics to user segments with different characteristics. We explore two factors for segmenting users: user activity frequency (number of user interactions) and category-wise entropy in the user profile. We simply used a percentile-based approach, segmenting users into three groups based on each factor. 
