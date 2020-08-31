import pandas as pd
import numpy as np
import pandasql
import pickle
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Represents columns, mlbid, start_speed, pfx_x, pfx_z, spinrateND, spindirND, pitch_type
train_cols = [2, 5, 10, 11, 15, 16, 21]

# Represents columns mlbid, start_speed, pfx_z, pfx_z, spinrateND, spindirND
predict_cols = [2, 5, 10, 11, 15, 16]

# Load in training data and supplemental data
train_pitches = pd.read_csv('sheets/train.csv', usecols=train_cols, header=0)
predict_pitches = pd.read_csv('sheets/test.csv', usecols=predict_cols, header=0)
players = pd.read_excel('sheets/players.xlsx', usecols=[7, 10, 27], header=0)
master = pd.read_csv('sheets/master.csv', usecols=[0, 2, 6])

# Query supplemental data to filter to just pitchers with mlbid's in the range of our training data
pitchers1 = pandasql.sqldf("SELECT mlb_id AS mlbid, throws FROM master WHERE mlb_pos = 'P' AND mlbid <= 621212 AND mlbid >= 110683")
pitchers2 = pandasql.sqldf("SELECT MLBID AS mlbid, THROWS AS throws FROM players WHERE POS = 'P' AND mlbid <= 621212 AND mlbid >= 110683")

# Join multiple data sources that include potential matches for the training dataset
pitchers = pandasql.sqldf("SELECT pitchers1.mlbid, pitchers1.throws FROM pitchers1 LEFT JOIN pitchers2 on pitchers1.mlbid = pitchers2.mlbid union SELECT pitchers2.mlbid, pitchers2.throws FROM pitchers2 LEFT JOIN pitchers1 on pitchers2.mlbid = pitchers2.mlbid WHERE pitchers2.mlbid IS NULL")
print (pitchers)


# Join training data with supplemental data and break into left handed group and right handed group
joined = pandasql.sqldf("SELECT * FROM train_pitches JOIN pitchers ON train_pitches.mlbid = pitchers.mlbid WHERE pitch_type = 'CH' OR pitch_type = 'CU' OR pitch_type = 'FA' OR pitch_type = 'SL'")
#pred_join = pandasql.sqldf('SELECT * FROM predict_pitches JOIN pitchers ON predict_pitches.mlbid = pitchers.mlbid')
left = pandasql.sqldf("SELECT * FROM joined WHERE throws = 'L'")
right = pandasql.sqldf("SELECT * FROM joined WHERE throws = 'R'")
num = pandasql.sqldf("SELECT DISTINCT mlbid from right")

print(right)
print(num)

#pred_left = pandasql.sqldf("SELECT * from pred_join WHERE throws = 'L'")
#print(pred_left)

# Split left and right handed data sets into test and train
Xr = right.values[:,1:6]
yr = right.values[:,6]
Xl = left.values[:,1:6]
yl = left.values[:,6]

Xr_train, Xr_validation, Yr_train, Yr_validation = train_test_split(Xr, yr, test_size=0.20, random_state=1)
Xl_train, Xl_validation, Yl_train, Yl_validation = train_test_split(Xl, yl, test_size=0.20, random_state=1)

# Train the model on left and right handed sets
model = LinearDiscriminantAnalysis()
model.fit(Xr, yr)
model.fit(Xl, yl)
#kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#r_results = cross_val_score(model, Xr_train, Yr_train, cv=kfold, scoring='accuracy')
#L_results = cross_val_score(model, Xl_train, Yl_train, cv=kfold, scoring='accuracy')
#print('%f (%f)' % (r_results.mean(), r_results.std()))

right_predictions = model.predict(Xr_validation)
print(accuracy_score(Yr_validation, right_predictions))
print(classification_report(Yr_validation, right_predictions))

left_prediction = model.predict(Xl_validation)
print(accuracy_score(Yl_validation, left_prediction))
print(classification_report(Yl_validation, left_prediction))