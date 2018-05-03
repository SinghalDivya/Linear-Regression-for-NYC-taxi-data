from pyspark.ml.regression import LinearRegression
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
import time
import pandas as pd
from datetime import datetime
from sklearn import linear_model
import datetime as dt

spark = SparkSession \
        .builder \
        .appName(<LinearRegression>) \
        .getOrCreate()

sc = spark.sparkContext
#start time
start = dt.datetime.now()
#question 1: Using Spark MLlib build a model to predict taxi fare from trip distance (M1)

training=sc.textFile("nyc_taxi.csv")
tmphead=training.first()
finalhead=tmphead.strip().split(',')
training=training.filter(lambda x: tmphead not in x)
training=training.map(lambda x: [k for k in  x.strip().split(',')])
training=training.map(lambda x: Row(label =int(float(x[6])),features=Vectors.dense(float(x[4]))))

trainingDF=spark.createDataFrame(training)

lr = LinearRegression()
lrModel = lr.fit(trainingDF)

#question 2 (1) : What is the fare of a 20 mile long trip using M1
new_data=[20]
testdata=sc.parallelize([Row(features=Vectors.dense(new_data))])
testdataDF=testdata.toDF()
testdata=lrModel.transform(testdataDF).head()
print testdata.prediction

end = dt.datetime.now()

print end-start


##### scikit learn model m1 to predict taxi fare from trip distance (M1)
from sklearn import linear_model
import datetime as dt

#start time
start = dt.datetime.now()


#Reading the data
data =  [k.strip().split(',') for k in open('nyc_taxi.csv','r').readlines()[1:]]

#Converting the features into float and putting them in a list
# Model M1
feature = []
label = []
for d in data:
	feature.append([float(d[-3])])
	label.append(float(d[-1]))

#Training
reg = linear_model.LinearRegression()
reg.fit (feature, label)

#predict 2 mile trip fare
print reg.predict([20])

#end time
end = dt.datetime.now()

print end-start

### question 3: Using Spark operations (transformation and actions) compute the average tip amount
training=sc.textFile("nyc_taxi.csv")
tmphead=training.first()
finalhead=tmphead.strip().split(',')
training=training.filter(lambda x: tmphead not in x)
training=training.map(lambda x: [k for k in  x.strip().split(',')])

train=training.map(lambda p: Row(Pickup_date=p[0],pickup_time=p[1],dropoff_date=p[2],dropoff_time=p[3],distance=p[4],tip=p[5],fare=p[6]))
schemadata=spark.createDataFrame(train)
type(schemadata)
schemadata.createOrReplaceTempView("datatable")
#schemadata.describe().show()
schemadata.show()
### average of tip ##
schemadata.describe('tip').show()
spark.sql("select AVG(tip) from datatable GROUP BY tip").show(1)
### question 4:  During which hour the city experiences the most number of trips?

spark.sql("SELECT distinct pickup_time FROM datatable").show(10)
spark.sql("SELECT * FROM (SELECT split(pickup_time,':')[0] as hour, count(split(pickup_time,':')[0]) as trip_count FROM datatable GROUP BY split(pickup_time,':')[0]) AS A ORDER BY trip_count DESC")\
.show(1)

print "Maximum number of trips in hour = 17"




