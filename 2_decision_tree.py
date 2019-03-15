from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
import os

classifier = tree.DecisionTreeClassifier("entropy")
vectorizer = DictVectorizer(sparse = False)

d_training = [
{'packed':1, 'contains_encrypted':0}, {'packed':0, 'contains_encrypted':0},
{'packed':0, 'contains_encrypted':0}, {'packed':1, 'contains_encrypted':1},
{'packed':1, 'contains_encrypted':0}, {'packed':0, 'contains_encrypted':1},
{'packed':1, 'contains_encrypted':0}, {'packed':0, 'contains_encrypted':0},]
v_label = [1, 1, 1, 1, 0, 0, 0, 0]

vectorizer.fit(d_training)

X = vectorizer.transform(d_training)
y = v_label

classifier.fit(X, y)

with open("classifier.dot", "w") as output_file:
    tree.export_graphviz(
        classifier,
        feature_names=vectorizer.get_feature_names( ),out_file = output_file)

os.system("dot classifier.dot -Tpng -o classifier.png")

d_test = {'packed':1,'contains_encrypted':0}
d1_test = {'packed':1,'contains_encrypted':1}
v_test = vectorizer.transform(d_test)
v1_test = vectorizer.transform(d1_test)

print(classifier.predict(v_test))
print(classifier.predict(v1_test))
