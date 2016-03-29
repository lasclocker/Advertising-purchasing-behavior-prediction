#encoding:utf-8
from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array
import time
t1 = time.ctime()
sc = SparkContext(appName="DianShang")
table1 = sc.textFile("/data/test/monitorData/*")
def f1(x):
	lt = x.split('^')
	userID = lt[0]
	return userID
result = table1.map(f1).distinct() #select userID and duplicate removal
user = result.collect()
#select feature: IMPNums
def f2(x):
	lt = x.split('^')
	userID = lt[0]
	if lt[9] == 'IMP':
		return userID
pairs = table1.map(f2).map(lambda s: (s, 1))
result3 = pairs.reduceByKey(lambda a,b: a + b)
result4 = result.map(lambda s:(s,1)).leftOuterJoin(result3) #与IMP次数左连接
#select feature: CLKNums
def f3(x):
	lt = x.split('^')
	userID = lt[0]
	if lt[9] == 'CLK':
		return userID
pairs = table1.map(f3).map(lambda s: (s, 1))
result5 = pairs.reduceByKey(lambda a,b: a + b)
result6 = result4.leftOuterJoin(result5) #result4 leftOuterJoin data of CLKNums
result6 = result6.map(lambda x:"(%s,%s)"%(x[0],x[1]))  #Very important!!
result6.saveAsTextFile('/user/team322/junli_trueTestFeature') #result6 is tuple,before saveAsTextFile,must be translated into string
t2 = time.ctime()
print 'Starting:\t%s\nEnding:\t%s'%(t1,t2)
