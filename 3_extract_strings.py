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

tmp_array = []
tmp_obj = {}
nb = 0
cur_directory = 1
limit = -1
current = 0
features_name = []

# Extract strings in files
for tmp_dir in directories:
	print("directory : ", tmp_dir)
	print("| file read | attributes number |")
	print("|-----|---------|")
	onlyfiles = [f for f in listdir(path_to_data + tmp_dir) if isfile(join(path_to_data + tmp_dir, f))]
	# For each file in the current folder
	for file in onlyfiles:
		# Extraction of the strings
		result = subprocess.Popen(["./convert.sh", tmp_dir + "/" + file], stdout=subprocess.PIPE).communicate()[0]
		# Split the data in a list
		result = result.decode("ascii").split('@')
		# For each string in the current file
		for elem in result:
			# if the string already exist in the dictionnary
			if elem in tmp_obj:
				tmp_obj[elem] += 1
			else:
				tmp_obj[elem] = 1
		# Append the dictionnary of the file
		dictionnary.append(tmp_obj);
		print("| ", file, " | ", len(tmp_obj), " |")
		nb_attr_all += len(tmp_obj)
		# Clear tmp_obj
		tmp_obj = {}
		# Append the expected output relative to the folder ("malware" or "bnign")
		expected_output.append(tmp_output)
		current += 1
		# if the number of files reach the limit
		if (current == cur_directory*limit):
			break
	tmp_output = 1;
	cur_directory += 1
#print("dictionnary : ", len(dictionnary))
print("dictionnary size : ", nb_attr_all)