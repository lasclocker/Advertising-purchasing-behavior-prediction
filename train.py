#encoding:utf-8
from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array
import time
t1 = time.ctime()
sc = SparkContext(appName="DianShang") #create a SparkContext object
table1 = sc.textFile("/data/train/monitorData/*") #create Text file RDDs
def f1(x):
	lt = x.split('^')
	userID = lt[0]
	return userID
result = table1.map(f1).distinct() #select userID and duplicate removal
user = result.collect() #bring them back to the driver program as a list of objects
table2 = sc.textFile("/data/train/transformData/*")
pairs = table2.map(lambda x: (x.split('^')[0],1)) #make label for transformData
result2 = result.map(lambda s:(s,1)).leftOuterJoin(pairs) #all users leftOuterJoin transformData
#select feature: IMPNums
def f2(x):
	lt = x.split('^')
	userID = lt[0]
	if lt[9] == 'IMP':
		return userID
pairs = table1.map(f2).map(lambda s: (s, 1)) #if user is IMP,record it as tuple:(user,1)
result3 = pairs.reduceByKey(lambda a,b: a + b)
result4 = result2.leftOuterJoin(result3) #result2 leftOuterJoin data of IMPNums
#select feature: CLKNums
def f3(x):
	lt = x.split('^')  
	userID = lt[0]
	if lt[9] == 'CLK':
		return userID
pairs = table1.map(f3).map(lambda s: (s, 1)) #if user is CLK,record it as tuple:(user,1)
result5 = pairs.reduceByKey(lambda a,b: a + b)
result6 = result4.leftOuterJoin(result5) #result4 leftOuterJoin data of CLKNums 
result6 = result6.map(lambda x:"(%s,%s)"%(x[0],x[1]))  #result6 is tuple,before saveAsTextFile,must be translated into string
result6.saveAsTextFile('/user/team322/junli_trainFeature') #Write the elements of the dataset as a text file

