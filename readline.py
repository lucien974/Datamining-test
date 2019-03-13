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

tmp_array = []
tmp_obj = {}
nb = 0
cur_directory = 1
limit = 5
current = 0

for tmp_dir in directories:
	print("directory : ", tmp_dir)
	onlyfiles = [f for f in listdir(path_to_data + tmp_dir) if isfile(join(path_to_data + tmp_dir, f))]
	for file in onlyfiles:
		result = subprocess.Popen(["./convert.sh", ""], stdout=subprocess.PIPE).communicate()[0]
		for elem in result:
			nb = result.count(elem)
			tmp_obj[elem] = nb;
		dictionnary.append(tmp_obj);
		expected_output.append(tmp_output)
		nb_attr_all += 58057
		if (current == cur_directory*limit):
			break
		current += 1
		print ("current : ", current)
	tmp_output = 1;
	cur_directory += 1

hashed = hasher.transform(d_training)
hashed = hashed.todense()
hashed = numpy.asarray(hashed)
hashed = hashed

vectorizer.fit(hashed)

X = hashed
y = v_label
classifier.fit(X, y)

with open("classifier.dot", "w") as output_file:
    tree.export_graphviz(
        classifier,
        feature_names=vectorizer.get_feature_names( ),out_file = output_file)

os.system("dot classifier.dot -Tpng -o classifier.png")