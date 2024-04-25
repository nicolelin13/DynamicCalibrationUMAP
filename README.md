# Dynamic of Calibration in Recommender System [UMAP'24]

Code for UMAP24 PAPER -- Beyond Static Calibration: The Impact of User Preference Dynamics on Calibrated Recommendation. We include data and code in folders as descripbed below. We also list detailed methodology description and key visualization results below. 

## Directory

**Data** 
* The pre-processed KuaiRec and GoodReads datasets are in the "Data" folder, each in the subfolder named after the dataset. Each dataset folder includes a series of datasets of sub-profiles within designed time windows. 

**Models**
* All the recommendations are built upon well-tuned Bayesian Personalized Ranking (BPR) and ItemKNN. The tuning and modeling used in this work is based on the implementation of RecBole (https://github.com/RUCAIBox/RecBole). The tuning of key parameters is based on an exhaustive search for the best NDCG@10. The model statistics, including the best selected tuned parameter and accuracy-based measurement (recall@10 and NDCG@10), are available in the "Model" folder. 

**Analysis**
* Two jupyter notebooks are provided in the folder "Analysis", one for KuaiRec and one for GoodReads. Meanwhile, all the informative visualizations created during the analysis are in the subfolder "Figures." 

## Methodology Details
Our experiment workflow is as shown in the following diagram (Figure 1). 

<img src="/Analysis/Figure/Updated Experiment Workflow.png">
<p align="center">Figure 1. Experiment Workflow</p>

### 1. Data prepparation
We use two datasets from distinct domains: the short video and the book, which are with the contrasting nature of user behavior. The short video platform is characterized by rapid user engagement and high interaction frequencies, where user preferences evolve quickly. For books, users tend to transition between different book genres more gradually. To better investigate the calibration dynamics with the shifts in preference patterns, we preprocess datasets to focus on active users across time windows of different sizes.

The time window selection over user profile is critical when creating users' subprofiles in Figure 1 part (a). In this exploratory work, we are interested in observing changes in calibration when extending the training time window sizes. Given that goal, we look for the smallest window size, leading to sufficient interactions for training and for representing user preferences. Due to differences in user interaction frequency and evolving preferences across domains, time window sizes vary across domains. Users in the book reading domain have less frequent interactions and more stable preferences than users in short video domain. After tuning the time window sizes for the two datasets (for KuaiRec, 1/7/14 days; for GoodReads, 3/6/12 months), we picked 1 day for KuaiRec and 0.5 years (6 months) for GoodReads.

#### 1.1 KuaiRec
KuaiRec dataset is mostly the original fully observed dataset. In the original dataset, KuaiRec small matrix includes 4,676,570 interactions between 1,411 users and 3,327 videos. 

- Preprocess 

Two basic preprocesses are applied: 1/ remove interactions without valid timestamps, 2/ create the binary preference signal attribute based on watch_ratio. After the full preprocess on KuaiRec, the full dataset is with 1,799,403 interactions between 1,411 users and 3,320 videos, and the data density is 0.62. 

Given the key research objective is to investigate the temporal calibration pattern, we exclude 181,992 interactions that are without valid timestamp values. 

The preference signal is watch_ratio, which is the play_duration divided by video_duration. The watch_ratio ranges from 0 to 571.52, where the mode is 0.77 and the mean is 0.91. As common sense applies, higher watch_ratio may indicate users like the given video. Referring to the mean of watch_ratio, we turned the signal into a binary by setting threshold value 0.9, where watch_ratio not lower than 0.9 is defined 1 and otherwise 0. 

- Sub dataset creation 

We tuned the time window size from 1 day to 14 dyas, and picked 1 day for KuaiRec. Based on the preprocessed full dataset, 63 sub datasets were created by accumulating days, starting from the first 1 day to all 63 days. 

#### 1.2 GoodReads
The original dataset that includes 2,360,655 books, 876,145 users, and 228,648,342 user-book interactions. Also, the books have been categorized in 8 genres. Books may overlap across different genres. 

- Preprocess 

To address the dual objectives of reducing the dataset's size and mitigating its high sparsity level we focused on interactions involving the most active users and popular items. The steps taken to achieve this were as follows: Firstly, the dataset was filtered to focus on active users defined as those with 10 to 20 interactions per year resulted to about 12M interactions with 99.9% sparsity. Then, we filtered out items with fewer than 1000 interactions. This step reduced the total number of interactions to about 3.5 million, and the sparsity decreased to 98%. Finally, we applied criteria to ensure that only users with a consistent level of activity (at least 5 interactions per year) were included in the dataset. The sparsity level decreased to about 93%.

- Sub dataset creation 

We tuned the time window size from 3 months to 12 months, and picked 6 months (0.5 years) for GoodReads. Based on the preprocessed full dataset, 16 sub datasets were created by accumulating 6 months, starting from the first 6 months to full 8 years. 


### 2. Simulation Process
We conducted a simulation-based analysis for identifying the most representative segments of users' profiles that, if used for training the recommendation model, would yield more calibrated recommendation results. 

(1) Given user-item interaction matrix containing the profile of all users and their interacted items, we sort each user $u$'s profile $P_u$ chronologically in descending order and split $P_u$ into $n$ subprofiles as $\{P_u^1, P_u^2, ..., P_u^n\}$ where $P_u^1$ contains the most recent $u$'s interactions and $P_u^n$ contains the oldest interactions. This process is shown in part (a) of Figure 1. The choice of $n$ (number of subprofiles) is a domain-specific parameter. The time *window size* for creating the subprofiles should be set based on how frequently users interact with items. For example, in a short video streaming platform where users interact with items very frequently, a time window of shorter length, like daily, may be appropriate. However, in a book recommendation platform, a time window of a longer period, such as one or more years, might better capture users' evolving tastes. We discuss and analyze the choice of the time window and $n$ on different recommendation domains in the methodology section.

(2) As shown in part (b) of Figure 1, we create samples of the dataset $D$ by iteratively and chronologically combining the subprofiles of users from different time windows as follows:
    $$D^l = \{P_u^1 \cup P_u^2 \cup ... \cup P_u^l | \forall u \in \mathcal{U}\}, \;\;\;\; where \;\; l \leq n$$
    This will create samples of the dataset (i.e., $D^1, D^2, ..., D^n$) as shown in part (c) of Figure 1. 

(3) We iteratively pick each sample created in the previous step and evaluate the recommendation performance built on that sample. As shown in part (d) of Figure 1, we first tune the recommendation model on each sample of the data for the best model. Then, in part (e) of Figure 1, with the best model identified in the previous step, we evaluate the recommendation performance (i.e., miscalibration and accuracy) on each sample of the dataset. 

When considering the different segments of the user's profile (from the most recent to the oldest interactions), the most recent segment of the users' profiles (i.e., $D^1$, $D^2$, ..., or $D^n$) that yields the lowest miscalibration is the one that is most representative of users' current preferences.

### 3. Other Key Experiment Basic Setup

To investigate the calibration pattern across different training time windows, we generate recommendations based on the training datasets in given time periods. Each recommendation is created via a specifically well-tuned model with the specific sub dataset. We applied two algorithms, BPR and ItemKNN, with tuning and training details listed below. Except for the miscalibration measurement, most of the experiments are conducted using the open-source recommendation library RecBole. 

#### 3.1 Model tuning 
With NDCG@10 as the objective, the hyperparameter tuning is via exhaustive search of given parameter ranges as provided below. Models for different time periods are independently tuned with the specific training set from the given time period. 
* BPR:
    - Learning rate: [0.01, 0.05, 0.001, 0.005, 0.0001]
    - Embedding size: [64, 96, 128]
* ItemKNN:
    - K (neighborhood size): [50, 100, 200]
    - Shrink (normalization hyper parameter in cosine distance calculation): [0, 0.5]

#### 3.2 Training and evaluation
For model training and evaluation with the well-tuned parameters, the dataset in a tested time period is randomly split into train, validation and test sets by the ratio 8:1:1. We focus on two metrics in this experiment, NDCG@10 (ranking accuracy) and miscalibration. NDCG@10 value is based on the value returned from RecBole, while miscalibration is calculated via the distribution comparison between the recommendation list and the preference shown in the given time period. 

Also, to reduce the random noise from experiments, we changed random seeds and repeated 6 times of the training and evaluation procedure mentioned above. The final evaluation results presented in the paper are based on the mean value from the repeated experiments.   

#### 4. User Segmentation
Given that users' varied behavioral patterns in the same dataset may be related to different degrees of calibration across individuals, we extend our analysis of calibration dynamics to user segments with different characteristics. We explore two factors for segmenting users: user activity frequency (number of user interactions) and category-wise entropy in the user profile. We simply used a percentile-based approach, segmenting users into three groups based on each factor. 

## Key Results (visualization) 
### 1. Baysian Personalization Ranking (BPR)
#### 1.1 Full population 
<img src="/Analysis/Figure/BPR/KuaiRec_FullMiscalibration.png">
<p align="center">Figure 1a. [KuaiRec BPR] Box plot of miscalibration distribution by time windows</p>

<img src="/Analysis/Figure/BPR/GoodReads_FullMiscalibration.png">
<p align="center">Figure 1b. [GoodReads BPR] Box plot of miscalibration distribution by time windows</p>

#### 1.2 Segmented users
* **Segment by user activity degree (number of overall interactions)**

<img src="/Analysis/Figure/BPR/KuaiRec_Activity.png">
<p align="center">Figure 1c. [KuaiRec BPR] Miscalibration by user segments by activity level</p>

<img src="/Analysis/Figure/BPR/GoodReads_Activity.png">
<p align="center">Figure 1d. [GoodReads BPR] Miscalibration by user segments by activity level</p>

* **Segment by user profile entropy (full user profile diversity measured by entropy, higher is more diversified)**

<img src="/Analysis/Figure/BPR/KuaiRec_ProfileEntropy.png">
<p align="center">Figure 1e. [KuaiRec BPR] Miscalibration by user segments by profile entropy</p>

<img src="/Analysis/Figure/BPR/GoodReads_ProfileEntropy.png">
<p align="center">Figure 1f. [GoodReads BPR] Miscalibration by user segments by profile entropy</p>

### 2. ItemKNN 
#### 2.1 Full population 
<img src="/Analysis/Figure/ItemKNN/KuaiRec_FullMiscalibrationItemKNN.png">
<p align="center">Figure 2a. [KuaiRec ItemKNN] Box plot of miscalibration distribution by time windows</p>

<img src="/Analysis/Figure/ItemKNN/GoodReads_FullMiscalibrationItemKNN.png">
<p align="center">Figure 2b. [GoodReads ItemKNN] Box plot of miscalibration distribution by time windows</p>

#### 1.2 Segmented users
* **Segment by user activity degree (number of overall interactions)**

<img src="/Analysis/Figure/ItemKNN/KuaiRec_ActivityDegreeItemKNN.png">
<p align="center">Figure 2c. [KuaiRec ItemKNN] Miscalibration by user segments by activity level</p>

<img src="/Analysis/Figure/ItemKNN/GoodReads_ActivityDegreeItemKNN.png">
<p align="center">Figure 2d. [GoodReads ItemKNN] Miscalibration by user segments by activity level</p>

* **Segment by user profile entropy (full user profile diversity measured by entropy, higher is more diversified)**

<img src="/Analysis/Figure/ItemKNN/KuaiRec_ProfileEntropyItemKNN.png">
<p align="center">Figure 2e. [KuaiRec ItemKNN] Miscalibration by user segments by profile entropy</p>

<img src="/Analysis/Figure/ItemKNN/GoodReads_ProfileEntropyItemKNN.png">
<p align="center">Figure 2f. [GoodReads ItemKNN] Miscalibration by user segments by profile entropy</p>

