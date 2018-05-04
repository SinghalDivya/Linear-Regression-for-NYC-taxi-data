#////////////////////////////////////////////////////////////////////////////////////////////////////
#/// \scikit-Learn.py
#/// \brief A python program built to build a model to predict taxi fare from trip distance.
#///
#//  Author: Divya Singhal
#////////////////////////////////////////////////////////////////////////////////////////////////////

#import the packages
from sklearn import linear_model
import datetime as dt

#start time
start = dt.datetime.now()

#Reading the data
data =  [k.strip().split(',') for k in open('nyc_taxi.csv','r').readlines()[1:]]

#Converting the features into float and putting them in a list
feature = []
label = []
for d in data:
	feature.append([float(d[-3])])
	label.append(float(d[-1]))

#Training the model 
reg = linear_model.LinearRegression()
reg.fit (feature, label)

#predict 2 mile trip fare
print reg.predict([20])

#end time
end = dt.datetime.now()

print end-start
