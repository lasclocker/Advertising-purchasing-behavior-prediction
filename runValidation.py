#encoding:utf-8
from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array
import time

t1 = time.ctime()
sc = SparkContext(appName="DianShang")
table1 = sc.textFile("/user/team322/junli_testFeature/*")
def f1(line):
	line = str(line).replace('(','').replace(')','').replace('None','0')
	userID = line.split(',')[0]
	return userID
user = table1.map(f1).collect() #select the users of validation data
result6 = sc.textFile("/user/team322/junli_trainFeature/*")
# Load and parse the data
def parsePoint(line):
	line = str(line).replace('(','').replace(')','').replace('None','0')
	line = line.split(',')
	values = [float(x) for x in line[2:]] #select label Column and features Columns 
	return LabeledPoint(values[0], values[1:])
parsedData = result6.map(parsePoint)
# Build the model
model = LogisticRegressionWithSGD.train(parsedData)
result7 = sc.textFile("/user/team322/junli_testFeature/*")
def testParsePoint(line):
	line = str(line).replace('(','').replace(')','').replace('None','0')
	line = line.split(',')
	values = [float(x) for x in line[1:]] #select label Column and features Columns
	return LabeledPoint(values[0], values[1:])
parsedData2 = result7.map(testParsePoint)
preds = parsedData2.map(lambda p: model.predict(p.features)) #use the model to predict parsedData2
preds = preds.collect() #translate the result of predict into list
userID = []
for i in xrange(len(preds)): #select users whose predict is 1
	if preds[i] == 1:
		userID.append(user[i])
sc.parallelize(userID).saveAsTextFile('/user/team322/solution_v') #create a parallelized collection and save it 
t2 = time.ctime()
print 'Starting:\t%s\nEnding:\t%s'%(t1,t2)