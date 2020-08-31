# Will Spencer - Nationals Baseball Pitch Classification Algorithm
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

train_cols = [5, 10, 11, 12, 15, 16, 21]#19, 20, 21]
predict_cols = [5, 10, 11, 12, 15, 16]#, 19, 20]

# Load in relevant data from the file and specify its type.
train_pitches = pd.read_csv('sheets/train.csv', usecols= train_cols)

# Split training data into left handed and right handed pitchers
pitches_right = pandasql.sqldf("SELECT start_speed, pfx_x, pfx_z, spinrateND, spindirND, pitch_type FROM train_pitches WHERE pitch_type = 'CH' OR pitch_type = 'CU' OR pitch_type = 'FA' OR pitch_type = 'SI' OR pitch_type = 'SL' OR pitch_type = 'KN'")
#pitches_left = pandasql.sqldf("SELECT start_speed, pfx_x, pfx_z, spinrateND, spindirND, xangle, zangle, pitch_type FROM train_pitches WHERE stand = 'L' AND pitch_type = 'CH' OR pitch_type = 'CU' OR pitch_type = 'FA' OR pitch_type = 'SI' OR pitch_type = 'SL' OR pitch_type = 'KN'")

# Set up training data for right-handed pitchers
right_array= pitches_right.values
Xr = right_array[:,0:5]
yr = right_array[:,5]

# Set up training data for left-handed pitchers
#left_array= pitches_left.values
#Xl = left_array[:,0:7]
#yl = left_array[:,7]

# Split left and right handed data sets into validation sets and training sets. 20% of each set will be used as a validation set
Xr_train, Xr_validation, Yr_train, Yr_validation = train_test_split(Xr, yr, test_size=0.20, random_state=1)
#Xl_train, Xl_validation, Yl_train, Yl_validation = train_test_split(Xl, yl, test_size=0.20, random_state=1)

# Apply each model to training data and print the result
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

# Evaluate on right handed dataset and store results
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, Xr_train, Yr_train, cv=kfold, scoring = 'accuracy')
print(results)
     
# Evaluate on left handed dataset and store results
#left_results = cross_val_score(model, Xl_train, Yl_train, cv=kfold, scoring = 'accuracy')
#left.append(left_results)
#names.append(name)

right_model = LinearDiscriminantAnalysis()
#left_model = LinearDiscriminantAnalysis()
right_model.fit(Xr_train, Yr_train)
#left_model.fit(Xl_train, Yl_train)

# Save the model to a pickle file to be used later
filename = open('model', 'wb')
pickle.dump(right_model, filename)
filename.close()

#filename = open('left_model', 'wb')
#pickle.dump(left_model, filename)
#filename.close()

# Fit the model to right handed training sets and run predictions
right_predictions = right_model.predict(Xr_validation)
print(accuracy_score(Yr_validation, right_predictions))
print(classification_report(Yr_validation, right_predictions))

#Fit the model to left handed training sets and run predictions
#left_predictions = left_model.predict(Xl_validation)
#print(accuracy_score(Yl_validation, left_predictions))
#print(classification_report(Yl_validation, left_predictions))

# Load in new dataset to make predictions on
predict_pitches = pd.read_csv('sheets/test.csv', usecols= predict_cols, header=0)
right = pandasql.sqldf("SELECT start_speed, pfx_x, pfx_z, spinrateND, spindirND FROM predict_pitches")
#left = pandasql.sqldf("SELECT start_speed, pfx_x, pfx_z, spinrateND, spindirND, xangle, zangle FROM predict_pitches WHERE stand = 'L'")

# Load in models and run prediction
right_model = pickle.load(open('right_model', 'rb'))
#left_model = pickle.load(open('left_model', 'rb'))
#l_pred = left_model.predict(left)
r_pred = right_model.predict(right)

#lf= pd.DataFrame(l_pred)
rf= pd.DataFrame(r_pred)

#lf.to_csv('left_predicted.csv')
rf.to_csv('right_predicted.csv')

#print(l_pred)
print(r_pred)