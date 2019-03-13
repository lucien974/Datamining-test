from os import listdir
from os.path import isfile, join
import subprocess
from sklearn.feature_extraction import FeatureHasher

path_to_data = "samples/"
directories = ["benignware", "challenge", "malware"]
#mypath = input("path : ")

#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

nb_attr = 0
nb_attr_all = 0
increment = 0
prev = 0
loop = 1
dictionnary = {}

#print(onlyfiles)
for tmp_dir in directories:
	print("directory : ", tmp_dir)
	onlyfiles = [f for f in listdir(path_to_data + tmp_dir) if isfile(join(path_to_data + tmp_dir, f))]
	for file in onlyfiles:
		#data = open(path_to_data + tmp_dir + "/" + file, "r")
		#result = subprocess.Popen(["strings", path_to_data + tmp_dir + "/" + file], stdout=subprocess.PIPE).communicate()[0]
		result = subprocess.Popen(["./convert.sh", ""], stdout=subprocess.PIPE).communicate()[0]
		#print(result)
		#print(file)
		#prev
		#for line in result:
		#	nb_attr += 1
		dictionnary += result;
			#print(line)
		#print("| ", increment, " | ", nb_attr, " |")
		#print("| ")
		#print(increment)
		#print(" | ")
		#print(nb_attr)
		#print (" |")
		#if (increment == 33*loop):
		#	loop += 1
		#	break
		nb_attr_all += 58057
		#prev = nb_attr
		#nb_attr = 0
		#increment += 1
		#break
	#break
#print(nb_attr_all)
#file = open(“testfile.txt”, “r”) 
#for line in file: 
#print line