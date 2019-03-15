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
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction import DictVectorizer

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
limit = 150
current = 0
features_name = []

def train_classifier(training_data, training_labels, testing_data, testing_labels, save):
	# Train the classifier
	X = training_data
	y = training_labels
	randforest.fit(X, y)

	# Save the classifier with pickle
	#with open('classifier_strings','wb') as fp:
	#    pickle.dump(classifier,fp)

	# Initialisation for the ROC curve
	fpr = []
	tpr = []

	# Compute fpr and tpr of the classifier
	result = randforest.predict(testing_data)
	fpr, tpr, thresholds = metrics.roc_curve(testing_labels, result)
	auc = metrics.roc_auc_score(testing_labels, result)
	#print("fpr : ", fpr)
	#print("tpr : ", tpr)

	# Show the ROC curve of the classifier
	pyplot.title('Receiver Operating Characteristic')
	pyplot.plot(fpr, tpr, label='ROC curve (area = %0.3f), split %s' %(auc, save))
	pyplot.legend(loc = 'lower right')
	pyplot.plot([0, 1], [0, 1],'r--')
	pyplot.xlim([0, 1])
	pyplot.ylim([0, 1])
	pyplot.ylabel('True Positive Rate')
	pyplot.xlabel('False Positive Rate')
	#pyplot.show()
	#pyplot.savefig(save + '.png')

# Extract strings in files
for tmp_dir in directories:
	print("directory : ", tmp_dir)
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

# Hash the dictionnary
hashed = hasher.transform(dictionnary)
hashed = hashed.todense()
hashed = numpy.asarray(hashed)
#print("hashed : ", hashed)

# Shuffle the data
tmp_list = list(zip(hashed, expected_output))
random.shuffle(tmp_list)
hashed, expected_output = zip(*tmp_list)

#print("cut : ", cut, ", len : ", len(hashed))

kf = KFold(n_splits=10)
hashed = numpy.array(hashed)
expected_output = numpy.array(expected_output)
for train_index, test_index in kf.split(hashed):
	#print("train_index : ", train_index, ", test_index : ", test_index)
	X_train, X_test = hashed[train_index], hashed[test_index]
	y_train, y_test = expected_output[train_index], expected_output[test_index]
	train_classifier(X_train, y_train, X_test, y_test, "x-validation")
pyplot.savefig('cross_validation.png')