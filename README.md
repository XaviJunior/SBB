# Team SBB

## Members
* Xavier AEBY
* Tarik BACHA
* Tanguy BERGUERAND
* Frederic SPYCHER

## Context

In the scope of the **Big Scale Analytics** course at HEC Lausanne, Team SBB is happy to present its two projects.

## Project 1: Explainable Artificial Intelligence (XAI)

The goal of this project is to introduce the subject of XAI and to exemplify it by showcasing several Python libraries: LIME, SHAP, ELI5, and XAI. We worked with data about Kickstarter used in a [previous project](https://github.com/tbacha/DMML2019_Team_Apple).

### Files

* `Team_SBB_project_1_Explainable_AI.ipynb`: main notebook, which contains a link to our YouTube video, an introduction about XAI, presentation + examples with code for each of the libraries mentioned above, and a conclusion.
  * **NB**: it is advised to run this notebook in *Google Colaboratory* for two reasons: 1) all package installations work properly, and 2) plots that rely on JavaScript are visible.
* For the notebook to run in Colab, the three files listed below must be uploaded manually. 
  * `cleaning.py`: cleaned version of the Kickstarter dataset
  * `custom_functions_SBB.py`: new version of the `custom_functions` module, updated for XAI purposes
  * `model_evaluation_utils.py`: useful module containing a mix of different performance metrics, by Dipanjan Sarkar (Google)

## Project 2: Real or Not? NLP with Disaster Tweets (Kaggle Competition)

The goal of this project is to train a model that can predict, using natural language processing (NLP), whether tweets about natural or man-made disasters are genuine.

### Files

#### Code

* `Team_SBB_project_2_Disaster_tweets.ipynb`: main notebook, which contains a link to our YouTube video; a description of the context and data; information about the cleaning, vectorization and training processes; model evaluation and final observations.
* `notebook_functions.py`: custom code that was separated from the main notebook to avoid unnecessary clutter (for the notebook to run in Colab, this file must be uploaded manually). 

#### Data

* `train.csv` / `test.csv`: datasets were provided by Kaggle. The former is classified and therefore used for training. The classes are missing from the latter and thus need to be predicted using the model.
* `cleaning_tests.xlsx`: results from some of the tests we performed to determine our cleaning strategy.
