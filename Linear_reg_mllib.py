#////////////////////////////////////////////////////////////////////////////////////////////////////
#/// \file Linear_reg_mllib.py
#/// \brief Linear regression model implementation using NYC Taxi database 
#///        - implementaion using Spark Machine learning library: Spark Mlib
#///
#//  Author: Divya Singhal
#////////////////////////////////////////////////////////////////////////////////////////////////////

### Importing the packages
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
#build the spark session
spark = SparkSession \
        .builder \
        .appName(<LinearRegression>) \
        .getOrCreate()

sc = spark.sparkContext
#set to the start time
start = dt.datetime.now()

#Load and parse the data
training=sc.textFile("nyc_taxi.csv")
tmphead=training.first()
finalhead=tmphead.strip().split(',')
training=training.filter(lambda x: tmphead not in x)
training=training.map(lambda x: [k for k in  x.strip().split(',')])
training=training.map(lambda x: Row(label =int(float(x[6])),features=Vectors.dense(float(x[4]))))
#create data frame for training model
trainingDF=spark.createDataFrame(training)
# Build the model 
lr = LinearRegression()
lrModel = lr.fit(trainingDF)

#Evaluating the model on test data
new_data=[20]
testdata=sc.parallelize([Row(features=Vectors.dense(new_data))])
testdataDF=testdata.toDF()
testdata=lrModel.transform(testdataDF).head()
print testdata.prediction

end = dt.datetime.now()
#print the total time taken to run the command.
print end-start
