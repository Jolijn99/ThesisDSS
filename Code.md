# Code thesis 
In this readme file the full code for my thesis can be found.
I started with loading all necessities. 

```
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
import csv
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
import imblearn
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import gc
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
!pip install pandas plotnine
import warnings
warnings.filterwarnings('ignore')
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, geom_histogram
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
from sys import path
import metrics
from sklearn.model_selection import GridSearchCV
```

Then I loaded my data from the ESS. Subsequentially I selected which variables I wanted to include in my study.
```
df = pd.read_csv("ESS10.csv", low_memory = False)
```

