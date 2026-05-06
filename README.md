# ML Ratings Predictor README

## This project analyzes audience and critic ratings for anime titles. A clustering model (using the scikit-learn Python library) is used to visualize trends and patterns across rating data.

Before running the project, install the required python libraries:
```pip install -r requirements.txt```

To run commands open command line to folder 'main-project':

```cd main-project```


To run default main.py, enter on command line:

```python main.py```


To manually add K for K clusters is

```Python main.py --k (Int # of cluster)```

EX: ```python main.py --k 15```

To run with best K enter (system will utilize a function to find the best k)

```Python main.py --best-k```


Optional:
to observe the data pipeline run ```Python main.py --raw``` to get the raw data. No visualizations will be made

to get the merged and loaded dataset run ```Python man.py --preprocessed``` to get the processed data that’ll be inputted into the model. No visualization will be made
