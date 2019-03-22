import os
import numpy
import subprocess
from os import listdir
from os.path import isfile, join
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher

path_to_data = "samples/"
directories = ["benignware", "malware"]
expected_output = []
tmp_output = 0
nb_attr_all = 0
dictionnary = []
classifier = tree.DecisionTreeClassifier("entropy")
vectorizer = DictVectorizer(sparse = False)

d_training = [{'packed':1, 'contains_encrypted':0}, {'packed':0, 'contains_encrypted':0},
{'packed':0, 'contains_encrypted':0}, {'packed':1, 'contains_encrypted':1}]
v_label = [1, 1, 0, 0]

hasher = FeatureHasher(n_features=2000)
#hashed = hashed[0]
def doAppend(size):
    result = []
    for i in range(size):
        
        result.append(0)
    return result

def hashing_vectorize(features,n):
    x = doAppend(n)
    for f in features:
        h = hash(f)
        i = h % n
        x[i] += 1
    return x

tmp_array = []
tmp_obj = {}
nb = 0
cur_directory = 1
limit = 25
current = 0
features_name = []

for tmp_dir in directories:
	print("directory : ", tmp_dir)
	onlyfiles = [f for f in listdir(path_to_data + tmp_dir) if isfile(join(path_to_data + tmp_dir, f))]
	for file in onlyfiles:
		result = subprocess.Popen(["./convert.sh", tmp_dir + "/" + file], stdout=subprocess.PIPE).communicate()[0]
		result = result.decode("ascii").split('@')
		#result = numpy.asarray(result)
		#if (isinstance(result, list) == False):
		#	break
		#print("result : ", result)
		#print("result[0] : ", result[0])
		for elem in result:
			#print("elem : ", elem)
			nb = result.count(elem)
			tmp_obj[elem] = nb;
		dictionnary.append(tmp_obj);
		expected_output.append(tmp_output)
		nb_attr_all += 58057
		if (current == cur_directory*limit):
			break
		current += 1
		#print ("current : ", current)
	tmp_output = 1;
	cur_directory += 1
data = []
dataNoHashed = [] 
for i in range(20):
    tab = []
    for f in dictionnary[i]:
        tab.append(f)
    dataNoHashed.append(tab)
    data.append(hashing_vectorize(dictionnary[i],20))
    #features_name.append(str(i))

for i in range(len(dataNoHashed)):
    print ("dataNoHashed: ",dataNoHashed[i])

for i in range(len(data)):
    print("dataHashed : ", data[i])

