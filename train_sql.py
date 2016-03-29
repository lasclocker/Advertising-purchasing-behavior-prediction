#encoding: utf-8
from pyspark import SparkContext
from operator import add
from pyspark.sql import SQLContext, Row
sc = SparkContext(appName="LYJDianShang")
sqlContext = SQLContext(sc)
# Load a text file.
lines = sc.textFile("/data/train/monitorData/part-00001")
def f1(x):
    if x=='IMP':
        return 1
    else:
        return 0    
def f2(x):
    if x=='CLK':
        return 1
    else:
        return 0
parts = lines.map(lambda l: l.split("^"))
#convert each line to a dictionary
record = parts.map(lambda p: Row(user=p[0],isValid=p[1], date=p[-2][0:8],IMP=int(f1(p[-1])),CLK=int(f2(p[-1]))))
# Infer the schema, and register the SchemaRDD as a table.
schemaRecord = sqlContext.inferSchema(record)
schemaRecord.registerTempTable("train_record")
# SQL can be run over SchemaRDDs that have been registered as a table.
trainData=sqlContext.sql("SELECT user,date,sum(IMP) as SUM_IMP, sum(CLK) as SUM_CLK FROM train_record where isValid='1' group by user,date ")
line1 = sc.textFile("/data/train/transformData/*")
part1 = line1.map(lambda l: l.split("^"))
label = part1.map(lambda p: Row(user=p[0],date=p[1][0:8])) # get train label
schemaLabel = sqlContext.inferSchema(label)
schemaLabel.registerTempTable("train_label")
trainData.registerTempTable("trainData")#noted:before use trainData to left outer join train_label,the trainData must be registered to a table
trainData1=sqlContext.sql("select a.user,a.date as date,SUM_IMP,SUM_CLK,b.date as label from trainData a left outer join train_label  b on (a.user=b.user and a.date=b.date)")
def f(x):
    if x=='None':
        return 0
    else:
        return 1
# The results of SQL queries are RDDs and support all the normal RDD operations.
Data=trainData1.map(lambda p:str(p.user) + '\t' + str(f(p.label)) + '\t' + str(p.SUM_IMP) + '\t' + str(p.SUM_CLK) + '\t' + str(p.date) )
Data.saveAsTextFile('/user/team322/yjlin__tsql_data')
