# Projet_TA


This project aims at tackling the problem of sentiment analysis of tweets in the context of the COVID-19 pandemic.

## Description

Natural Language Processing (NLP) is a very hot field these days, especially with social networks. In the IFT712 - Learning Techniques course at the University of Sherbrooke we will study this field applied to tweets. 

In concrete terms, the project selected is the classification of tweets on COVID-19. The complete dataset is available in the references at the end of the report.

The objective is therefore to produce several efficient classification algorithms. The objective will be achieved by different important steps such as visualization and data preprocessing.

Our database consists of five classes: Extremely Negative, Negative, Neutral, Positive and Extremely Positive. Our goal is to classify these tweets in order to know which type of sentiment it belongs to. We decided to keep only 3 classes: Negative, Neutral and Positive. 

## Getting Started

### Dependencies

* Libraries: requirements.txt
* Tested on Linux and Windows operating systems.

### Structure of the project

* Root : Where the main_notebook.ipynb and python scripts are located.
* notebooks_for_test : Where the notebooks for specific tests are located.
* out : Where the outputs of the visualizations outputs are located.

### Installing

* clone the repository:
* install the dependencies: 

```
pip install -r requirements.txt
```


### Executing program

* run the notebook main_notebook.ipynb

## Help

* Recommended to run in a google colab notebook :
    * Access to GPU
    * Better display

## Authors

Contributors names and contact info

CHANTRE Honorine  CHAH2807 : https://github.com/ChantreHonorine

THOMAS Eliott THOE2303 : https://github.com/eliottthomas99



## Acknowledgments

Inspiration, code snippets, etc.
* [original-dataset](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)
* [readme-template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
* [rnn-inspiration](https://www.kaggle.com/code/shahraizanwar/covid19-tweets-sentiment-prediction-rnn-85-acc)
* [tansformers](https://www.kaggle.com/code/ludovicocuoghi/twitter-sentiment-analysis-with-bert-roberta)




