import pandas as pd
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#from sklearn.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from matplotlib.legend_handler import HandlerLine2D
