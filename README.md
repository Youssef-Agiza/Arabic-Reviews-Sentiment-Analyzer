# Arabic-Reviews-Sentiment-Analyzer
A Neural Network model that predicts the sentiment(positive/negative) of arabic text achieving an accuracy of 83%. The dataset used to train the model is the [Arabic 100k reviews](https://www.kaggle.com/datasets/abedkhooli/arabic-100k-reviews) from Kaggle. The process of developing this model was divided on 5 phases, each has its own folder in the repo along with an attached report to explain what the phase includes.

We also built a user interface using flask to be able to try the app.



## Prerequisite to run the web app:
- Having python3
- Having pip and knowing how to start a virtual environment

## Usage

1. clone the repo\
```git clone https://github.com/Youssef-Agiza/Arabic-Reviews-Sentiment-Analyzer <dir_name>```
2. navigate to the following directory\
```cd <dir_name>/webApp ```
3. create the virtual environment if there isn't one yet\
```python3 -m venv env```
4. start the virtual environment\
```source env/bin/activate```
5. install the dependencies\
```pip install -r requirements```

6. run the app using:\
```python3 app.py```

7. Now you should be able to access the app on the localhost at http://127.0.0.1:5000
