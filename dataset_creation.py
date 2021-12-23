from library import *

"""
================================================================================================================================

Variables:

dataset = not normalized dataset

df_norm = normalized dataset

X = dataframe of features
y = dataset of target

X_big_array = np.array of the features (DictVectorize applied)

================================================================================================================================
"""


def norm_X(X_total):
    # transform the dataframe in dictionary to perform feature extraction
    X_total = X_total.to_dict('records')
  
    v = DictVectorizer(sparse = False, dtype = float)
    X_total = v.fit_transform(X_total)
    return X_total, v


dataset = pd.read_csv("healthcare-dataset-stroke-data.csv")

dataset.bmi.fillna(dataset.bmi.mean(), inplace=True)


df_norm = dataset.copy()

std = StandardScaler()
cols = ['age','avg_glucose_level', 'bmi']
norm = std.fit_transform(df_norm[cols])

df_norm[cols] = pd.DataFrame(norm)


# GENDER: F/M --> 1/0
df_norm.drop(df_norm.loc[dataset['gender'] =='Other'].index, inplace=True)
df_norm["gender"] = df_norm["gender"].apply(lambda x: 1 if x=="Female" else 0)

# EVER_MARRIED: YES/NO --> 1/0
df_norm["ever_married"] = df_norm["ever_married"].apply(lambda x: 1 if x=="Yes" else 0)

# RESIDENCE_TYPE: URBAN/RURAL --> 1/0
df_norm["Residence_type"] = df_norm["Residence_type"].apply(lambda x: 1 if x=="Urban" else 0)


X = df_norm.drop(['id', 'stroke'], axis = 1)
y = df_norm['stroke']



X_big_array = X.copy()
X_big_array, v = norm_X(X_big_array)


oversample = SMOTE()
X_res, y_res = oversample.fit_resample(X_big_array, y.array)