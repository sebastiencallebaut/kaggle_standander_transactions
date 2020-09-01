import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.utils import resample
import lightgbm as lgbm
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import catboost as cb
from catboost import Pool
from sklearn.model_selection import KFold

# Source
# https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
# https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18
# https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
# https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
# https://www.kaggle.com/mytymohan/sct-prediction-eda-smote-lgbm
# https://www.kaggle.com/wakamezake/starter-code-catboost-baseline

# Load the data sets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Reduce size of set to speed up our model build
#train_df = train_df.head(100)

# Create a merged data set and review initial information
combined_df = pd.concat([train_df, test_df])
print(combined_df.describe())
print(combined_df.info())

# Check missing values
print(combined_df.columns[combined_df.isnull().any()])

# No missing values and no exploration to conduct as we only have var_X

# Get the data types
print(Counter([combined_df[col].dtype for col in combined_df.columns.values.tolist()]).items())

# Set the ID col as index
for element in [train_df, test_df]:
    element.set_index('ID_code', inplace = True)

# Create X_train_df and y_train_df set
X_train_df = train_df.drop("target", axis = 1)
y_train_df = train_df["target"]

# Scale the data and use RobustScaler to minimize the effect of outliers
scaler = RobustScaler()

# Scale the X_train set
X_train_scaled = scaler.fit_transform(X_train_df.values)
X_train_df = pd.DataFrame(X_train_scaled, index = X_train_df.index, columns= X_train_df.columns)

# Scale the X_test set
X_test_scaled = scaler.transform(test_df.values)
X_test_df = pd.DataFrame(X_test_scaled, index = test_df.index, columns= test_df.columns)

# Split our training sample into train and test, leave 20% for test
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.2, random_state = 20)






# CLASS IMBALANCE

# Downsample majority class

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# Separate minority and majority classes
not_transa = X[X.target==0]
transa = X[X.target==1]

not_transa_down = resample(not_transa,
                                replace = False, # sample without replacement
                                n_samples = len(transa), # match minority n
                                random_state = 27) # reproducible results

# Combine minority and downsampled majority
downsampled = pd.concat([not_transa_down, transa])

# Checking counts
print(downsampled.target.value_counts())

# Create training set again
y_train = downsampled.target
X_train = downsampled.drop('target', axis=1)

print(len(X_train))



"""
# Upsample minority class

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# Separate minority and majority classes
not_transa = X[X.target==0]
transa = X[X.target==1]

not_transa_up = resample(transa,
                                replace = True, # sample without replacement
                                n_samples = len(not_transa), # match majority n
                                random_state = 27) # reproducible results

# Combine minority and downsampled majority
upsampled = pd.concat([not_transa_up, not_transa])

# Checking counts
print(upsampled.target.value_counts())

# Create training set again
y_train = upsampled.target
X_train = upsampled.drop('target', axis=1)

print(len(X_train))




# Create synthetic samples

sm = SMOTE(random_state=27, sampling_strategy='minority')
X_train, y_train = sm.fit_sample(X_train, y_train)

print(y_train.value_counts())

"""







"""
# OUTLIERS

# Remove outliers automatically
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)
print(yhat)

# Select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train.loc[mask, :], y_train.loc[mask]

print(len(X_train))
"""






# NEURAL NETWORK

# Build our neural network with input dimension 200
classifier = Sequential()

# First Hidden Layer
classifier.add(Dense(150, activation='relu', kernel_initializer='random_normal', input_dim=200))

# Second  Hidden Layer
classifier.add(Dense(350, activation='relu', kernel_initializer='random_normal'))

# Third  Hidden Layer
classifier.add(Dense(250, activation='relu', kernel_initializer='random_normal'))

# Fourth  Hidden Layer
classifier.add(Dense(50, activation='relu', kernel_initializer='random_normal'))

# Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

# Compile the network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

# Fitting the data to the training data set
classifier.fit(X_train,y_train, batch_size=10, epochs=200)

# Evaluate the model on training data
eval_model=classifier.evaluate(X_train, y_train)
print(eval_model)

# Make predictions on the hold out data
y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Get the accuracy score
print("Accuracy of {}".format(accuracy_score(y_test, y_pred)))

# Get the f1-Score
print("f1 score of {}".format(f1_score(y_test, y_pred)))

# Get the recall score
print("Recall score of {}".format(recall_score(y_test, y_pred)))

# Make predictions and create submission file
predictions = (classifier.predict(X_test_df)>0.5)
predictions = np.concatenate(predictions, axis=0 )
my_pred_ann = pd.DataFrame({'ID_code': X_test_df.index, 'target_ann': predictions})

# Set 0 and 1s instead of True and False
my_pred_ann["target_ann"] = my_pred_ann["target_ann"].map({True:1, False : 0})

# Create CSV file
my_pred_ann.to_csv('pred_ann.csv', index=False)




"""

"""
# LIGHT GBM

# Instantiate classifier

classifier = lgbm.LGBMClassifier(
    objective='binary',
    #metric='binary_logloss',
    metric = 'auc',
    boosting='gbdt',
    num_leaves=10,
    learning_rate=0.01,
    n_estimators=20000,
    #max_bin=50,
    max_bin=200,
    max_depth=-1,
    min_gain_to_split = 2,
    bagging_fraction=0.75,
    bagging_freq=5,
    bagging_seed=7,
    feature_fraction=0.5,
    feature_fraction_seed=7,
    verbose=-1,
    min_data_in_leaf=80,
    min_sum_hessian_in_leaf=11
)


# Fit the data
classifier.fit(X_train, y_train,)

# Make predictions on the hold out data
y_pred = (classifier.predict_proba(X_test)[:,1] >= 0.5).astype(int)

# Get the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Get the accuracy score
print("Accuracy of {}".format(accuracy_score(y_test, y_pred)))

# Get the f1-Score
print("f1 score of {}".format(f1_score(y_test, y_pred)))

# Get the recall score
print("Recall score of {}".format(recall_score(y_test, y_pred)))

# Make predictions
predictions = (classifier.predict_proba(X_test_df)[:,1] >= 0.5).astype(int)

# Create submission file
my_pred_lgbm = pd.DataFrame({'ID_code': X_test_df.index, 'target_lgbm': predictions})

# Create CSV file
my_pred_lgbm.to_csv('pred_lgbm.csv', index=False)







# XGBOOST

# Instantiate classifier

classifier = XGBClassifier(
    tree_method = 'hist',
    objective = 'binary:logistic',
    eval_metric = 'auc',
    learning_rate = 0.01,
    max_depth = 2,
    colsample_bytree = 0.35,
    subsample = 0.8,
    min_child_weight = 53,
    gamma = 9,
    silent= 1)

# Fit the data
classifier.fit(X_train, y_train)

# Make predictions on the hold out data
y_pred = (classifier.predict_proba(X_test)[:,1] >= 0.5).astype(int)

# Get the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Get the accuracy score
print("Accuracy of {}".format(accuracy_score(y_test, y_pred)))

# Get the f1-Score
print("f1 score of {}".format(f1_score(y_test, y_pred)))

# Get the recall score
print("Recall score of {}".format(recall_score(y_test, y_pred)))

# Make predictions
predictions = (classifier.predict_proba(X_test_df)[:,1] >= 0.5).astype(int)

# Create submission file
my_pred_xgb = pd.DataFrame({'ID_code': X_test_df.index, 'target_xgb': predictions})

# Create CSV file
my_pred_xgb.to_csv('pred_xgb.csv', index=False)









# CATBOOST

# Instantiate classifier
classifier = cb.CatBoostClassifier(loss_function="Logloss",
                           eval_metric="AUC",
                           learning_rate=0.01,
                           iterations=1000,
                           random_seed=42,
                           od_type="Iter",
                           depth=10,
                           early_stopping_rounds=500
                          )


# Fit the data
classifier.fit(X_train, y_train)

# Make predictions on the hold out data
y_pred = (classifier.predict_proba(X_test)[:,1] >= 0.5).astype(int)

# Get the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Get the accuracy score
print("Accuracy of {}".format(accuracy_score(y_test, y_pred)))

# Get the f1-Score
print("f1 score of {}".format(f1_score(y_test, y_pred)))

# Get the recall score
print("Recall score of {}".format(recall_score(y_test, y_pred)))

# Make predictions
predictions = (classifier.predict_proba(X_test_df)[:,1] >= 0.5).astype(int)

# Create submission file
my_pred_cat = pd.DataFrame({'ID_code': X_test_df.index, 'target_cat': predictions})

# Create CSV file
my_pred_cat.to_csv('pred_cat.csv', index=False)




# ENSEMBLE

# Create data frame
my_pred_ens = pd.concat([my_pred_ann, my_pred_xgb, my_pred_cat, my_pred_lgbm], axis = 1, sort=False)

# Review our frame
print(my_pred_ens.describe())

# Sum all the predictions and only assign a 1 if sum is higher than 2
my_pred_ens["target"] = my_pred_ens["target_ann"] + my_pred_ens["target_xgb"] + my_pred_ens["target_lgbm"] + my_pred_ens["target_cat"]

# Assign a 1 if sum is higher than 2
my_pred_ens["target"] = np.where(my_pred_ens["target"] > 2, 1, 0)

# Remove other target cols
my_pred_ens = my_pred_ens.drop(["target_ann", "target_lgbm", "target_xgb", "target_cat"], axis = 1)

# Create submission file
my_pred = pd.DataFrame({'ID_code': X_test_df.index, 'target': my_pred_ens["target"]})

# Create CSV file
my_pred.to_csv('pred_ens.csv', index=False)



"""



# KFOLD ON CATBOOST

n_split = 5
kf = KFold(n_splits=n_split, random_state=42, shuffle=True)

train_id = X_train.index
test_id = X_test_df.index

target = y_train
train = X_train
test = X_test_df

y_valid_pred = 0 * target
y_test_pred = 0


for idx, (train_index, valid_index) in enumerate(kf.split(train)):

    y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
    X_train, X_valid = train.iloc[train_index,:], train.iloc[valid_index,:]

    _train = Pool(X_train, label=y_train)
    _valid = Pool(X_valid, label=y_valid)

    print( "\nFold ", idx)

    fit_model = classifier.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=200,
                          early_stopping_rounds=1000
                         )

    pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  auc = ", roc_auc_score(y_valid, pred) )

    y_valid_pred.iloc[valid_index] = pred
    y_test_pred += fit_model.predict_proba(test)[:,1]

y_test_pred /= n_split


submission = pd.read_csv("sample_submission.csv")
submission['target'] = y_test_pred
submission['target'] = np.where(submission['target'] > 0.5, 1, 0)
submission.to_csv('submission_kfold.csv', index=False)






#preds_check['target'] = np.where(preds_check['target'] > 0.5, 1, 0)


# Find best threshold
"""


