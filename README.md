# Dynamic of Calibration in Recommender System [UMAP'24]

Code for UMAP23 PAPER -- Beyond Static Calibration: Dynamics of Calibration in Collaborated Recommender Systems

## Data
The pre-processed KuaiRec and GoodReads datasets are in the 'Data' folder, each in the subfolder named after the dataset. Each dataset folder includes a series of datasets of sub-profiles within designed time windows. 

## Models 
All the recommendations are built upon well-tuned Bayesian Personalized Ranking (BPR). The BPR used in this work is based on the implementation of RecBole (https://github.com/RUCAIBox/RecBole). The tuning of key parameters, embedding size, and learning rate is based on an exhaustive search for the best NDCG@10. 

## Analysis
Two jupyter notebooks are provided, one for KuaiRec and one for GoodReads. Meanwhile, all the informative visualizations created during the analysis are in the folder "Figures." 

