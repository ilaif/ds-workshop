
<img src="https://www.avito.ru/files/avito/company/logos/Logo-Avito.png" alt="Avito Logo" width="400px"/>

# Avito Demand Prediction Challenge
Our challenge is to predict demand for online ads by exposed features.

### [Project Documentation](./project_documentation.pdf)

## Dataset

* #### [Original Dataset from Kaggle](https://www.kaggle.com/c/avito-demand-prediction/data)
* #### [./helper_data Folder](https://drive.google.com/open?id=1GrepBq4JiV4LZ9lvslF8ygDzxrgg8WzD)
(Where we keep all of the intermediary outputs of extracted features, etc...)

## Data Exploration and Visualization

Extensive data exploration and visualization was conducted to understand the data better for feature engineering and strategy for choosing our algorithms toolkit.

#### [Data Exploration Notebook](./data_exploration.ipynb)

## Feature extraction/engineering

The feature engineering was composed of 4 main steps:
* #### Data Imputation (see below)
* #### [Image Feature Engineering Notebook](./feature_engineering/image_feature_engineering.ipynb)
* #### [Text Feature Engineering Notebook](./feature_engineering/nlp_feature_engineering.ipynb)
* #### [Categorical Feature Engineering](./feature_engineering/feature_enrichment.py)

### Data Imputation

We leveraged deep learning models to learn missing data of important features. We learned the important features only after this step, and then re-iterated the whole learning with the imputated data. We recommend reading those notebooks only after getting familiar with our Neural Network models (below).

#### [Imputating image_top_1 as continuous with NN notebook](./feature_engineering/NN-Stephan-LearnImageTop1-regression.ipynb)
#### [Imputating image_top_1 as a class with NN notebook](./feature_engineering/NN-Stephan-LearnImageTop1.ipynb)
#### [Imputating price with NN notebook](./feature_engineering/NN-Stephan-LearnPrice.ipynb)

***[feature_enrichment.py](./utils/feature_enrichment.py)*** module contains the loading functions for some of the resulted feature engineering work.

All outputs of the various feature engineering tasks are outputted as the following files and loaded as features before running the algorithms.

## ML Algorithms, Models and Model evaluation

The first algorithm family that comes to mind after exploring the data and extracting features is the **Decision Tree** family. Many algorithms are know in the field, while ensembling methods rule with superior performance and speed, between three popular decisions: *XGBoost*, *CatBoost* and *LGBM*, we finally chose **LGBM** after also testing *CatBoost* with lower performance (XGboost wasn't considered since it's inferior when dealing with heavy categorical data).

### Naive Learning

We first run some basic algorithms to produce naive results:

#### [Naive Learning Notebook](./algorithms/naive_learning.ipynb)

### CatBoost

CatBoost is a competitor of LGBM, but with inferior results as seen in the notebook

#### [CatBoost Notebook](./algorithms/catboost.ipynb)

### LGBM

We trained LGBM for a few iterations, and improved the features and hyperparameters in every iteration until we got our best score:

#### [LGBM - Final iteration Notebook](./algorithms/lgbm_final-0.2281.ipynb)

### Neural Networks and Deep Learning

LGBM is a great candidate here since there are many categorical features, but Deep Learning is a very strong family of classifiers and can many times produce better results.
As mentioned in the documentation, We had trained 10 neural networks:

TF-IDF text vectorization:

#### [NN-TFIDF-SEPARATED-UNIGRAMS](./NN/NN-Stephan-NN-TFIDF-SEPARATED-UNIGRAMS-nodropout.ipynb)

#### [NN-TFIDF-SEPARATED-BIGRAMS](./NN/NN-Stephan-NN-TFIDF-SEPARATED-BIGRAMS-nodropout.ipynb)

#### [NN-TFIDF-MERGED-UNI](./NN/NN-Stephan-NN-TFIDF-MERGED-UNI-nodropout.ipynb)

#### [NN-TFIDF-MERGED-BIGRAMS](./NN/NN-Stephan-NN-TFIDF-MERGED-BIGRAMS-nodropout.ipynb)

CountVec text vectorization:

#### [NN-CountVec-SEPARATED-UNIGRAMS](./NN/NN-Stephan-NN-CountVec-SEPARATED-UNIGRAMS-nodropout.ipynb)

#### [NN-CountVec-SEPARATED-BIGRAMS](./NN/NN-Stephan-NN-CountVec-SEPARATED-BIGRAMS-nodropout.ipynb)

#### [NN-CountVec-MERGED-UNI-nodropout](./NN/NN-Stephan-NN-CountVec-MERGED-UNI-nodropout.ipynb)

#### [NN-CountVec-MERGED-BIGRAM](./NN/NN-Stephan-NN-CountVec-MERGED-BIGRAMS-nodropout.ipynb)

FastText word embeddings + LSTM text processing (Trained on GCP, need ~50GB to run):

#### [NN-LSTM-MERGED-TEXT-COMBINED-NODROPOUT](./NN/NN-Stephan-LSTM-MERGED-TEXT-COMBINED-NODROPOUT.ipynb)

#### [NN-LSTM-MERGED-TEXT-COMBINED](./NN/NN-Stephan-LSTM-MERGED-TEXT-COMBINED.ipynb)

#### [NN-JUST-LSTM-MERGED-NODROPOUT](./NN/NN-Stephan-JUST-LSTM-MERGED-NODROPOUT.ipynb)


### Ensembling
For the general ensembling we've use two techniques, a meta-network (for non linear ensembling) and a lasso regression model (for linear ensembling). Those get as input the predictions of all other models and give the final prediction as an output.

#### [The general ensemble](./NN/NN-Stephan-Ensemble.ipynb)
