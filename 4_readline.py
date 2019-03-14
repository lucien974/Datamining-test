import os
import numpy
import subprocess
from os import listdir
import pickle as pickle
from os.path import isfile, join
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import RandomForestClassifier

path_to_data = "samples/"
directories = ["benignware", "malware"]
expected_output = []
tmp_output = 0
nb_attr_all = 0
dictionnary = []
classifier = tree.DecisionTreeClassifier("entropy")
randforest = RandomForestClassifier(n_jobs=2)
vectorizer = DictVectorizer(sparse = False)

d_training = [{'packed':1, 'contains_encrypted':0}, {'packed':0, 'contains_encrypted':0},
{'packed':0, 'contains_encrypted':0}, {'packed':1, 'contains_encrypted':1}]
v_label = [1, 1, 0, 0]

hasher = FeatureHasher(n_features=2000)
#hashed = hashed[0]

tmp_array = []
tmp_obj = {}
nb = 0
cur_directory = 1
limit = 5
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

#for i in range(2000):
#	features_name.append(str(i));


#print("data : ", dictionnary)

hashed = hasher.transform(dictionnary)
hashed = hashed.todense()
hashed = numpy.asarray(hashed)
hashed = hashed

#print("hashed : ", hashed)

#vectorizer.fit(hashed)

X = hashed
y = expected_output
randforest.fit(X, y)
#classifier.fit(X, y)

with open('classifier_strings','wb') as fp:
    pickle.dump(classifier,fp)

tmp_obj = {}
tf_do_i_need_this = []

test_files = [f for f in listdir(path_to_data + "challenge/") if isfile(join(path_to_data + "challenge/", f))]
for file in test_files:
	print("Extraction of ", file, " strings")
	result = subprocess.Popen(["./convert.sh", "challenge/" + file], stdout=subprocess.PIPE).communicate()[0]
	result = result.decode("ascii").split('@')
	for elem in result:
		nb = result.count(elem)
		tmp_obj[elem] = nb;

	#print("[debug] tmp_obj : ", tmp_obj)
	print("Hashing of the data of ", file)
	tf_do_i_need_this.append(tmp_obj)
	tmp_hash = hasher.transform(tf_do_i_need_this)
	tmp_hash = tmp_hash.todense()
	tmp_hash = numpy.asarray(tmp_hash)

print("results : ", randforest.predict(tmp_hash))
#with open("classifier.dot", "w") as output_file:
#    tree.export_graphviz(
#        classifier,
#        feature_names=features_name,out_file = output_file)

#os.system("dot classifier.dot -Tpng -o classifier.png")