import os
import numpy
import random
import subprocess
from os import listdir
import pickle as pickle
from os.path import isfile, join
from sklearn import tree
from sklearn import metrics
from matplotlib import pyplot
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
hasher = FeatureHasher(n_features=2000)

d_training = [{'packed':1, 'contains_encrypted':0}, {'packed':0, 'contains_encrypted':0},
{'packed':0, 'contains_encrypted':0}, {'packed':1, 'contains_encrypted':1}]
v_label = [1, 1, 0, 0]

tmp_array = []
tmp_obj = {}
nb = 0
cur_directory = 1
limit = 5
current = 0
features_name = []

# Extract strings in files
for tmp_dir in directories:
	print("directory : ", tmp_dir)
	onlyfiles = [f for f in listdir(path_to_data + tmp_dir) if isfile(join(path_to_data + tmp_dir, f))]
	for file in onlyfiles:
		result = subprocess.Popen(["./convert.sh", tmp_dir + "/" + file], stdout=subprocess.PIPE).communicate()[0]
		result = result.decode("ascii").split('@')
		for elem in result:
			if elem in tmp_obj:
				tmp_obj[elem] += 1
			else:
				tmp_obj[elem] = 1
		dictionnary.append(tmp_obj);
		expected_output.append(tmp_output)
		nb_attr_all += 58057
		if (current == cur_directory*limit):
			break
		current += 1
	tmp_output = 1;
	cur_directory += 1

# Hash the dictionnary
hashed = hasher.transform(dictionnary)
hashed = hashed.todense()
hashed = numpy.asarray(hashed)
hashed = hashed

# Shuffle the data
tmp_list = list(zip(hashed, expected_output))
random.shuffle(tmp_list)
hashed, expected_output = zip(*tmp_list)
cut = int(0.8*len(hashed))

# Divide the data
training_data = hashed[:cut]
training_labels = expected_output[:cut]
testing_data = hashed[cut:]
testing_labels = expected_output[cut:]

# Train the classifier
X = training_data
y = training_labels
randforest.fit(X, y)

# Save the classifier with pickle
with open('classifier_strings','wb') as fp:
    pickle.dump(classifier,fp)

# Initialisation for the challenge
tmp_obj = {}
tf_do_i_need_this = []

# Predict the challenge data
test_files = [f for f in listdir(path_to_data + "challenge/") if isfile(join(path_to_data + "challenge/", f))]
for file in test_files:
	print("Extraction of ", file, " strings")
	result = subprocess.Popen(["./convert.sh", "challenge/" + file], stdout=subprocess.PIPE).communicate()[0]
	result = result.decode("ascii").split('@')
	for elem in result:
		#nb = result.count(elem)
		if elem in tmp_obj:
			tmp_obj[elem] += 1
		else:
			tmp_obj[elem] = 1

	#print("[debug] tmp_obj : ", tmp_obj)
	print("Hashing of the data of ", file)
	tf_do_i_need_this.append(tmp_obj)
	tmp_hash = hasher.transform(tf_do_i_need_this)
	tmp_hash = tmp_hash.todense()
	tmp_hash = numpy.asarray(tmp_hash)

# Print the results
print("results : ", randforest.predict(tmp_hash))
