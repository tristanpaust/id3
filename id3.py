#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
import math
# Node class for the decision tree
import node

train=None
varnames=None
test=None
testvarnames=None
root=None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
	divided = p
	if divided <= 0 or 1-divided <= 0:
		return 0
	if divided == 1-divided:
		return 1
	if divided > 0 and 1-divided > 0:
		entropy = -divided * math.log((divided),2) - (1-divided)* math.log((1-divided),2)
		return entropy

# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):	

	if py_pxi == 0 or pxi == 0:
		positive_entropy = 0
	else:
		pos = (py_pxi+0.0)/pxi
		positive_entropy = entropy(pos)
	if py-py_pxi == 0:
		negative_entropy = 0
	else:
		neg = ((py-py_pxi)+0.0)/(total-pxi)
		negative_entropy = entropy(neg)
	
	total_entropy = (py+0.0)/total
	
	entropy_total = entropy(total_entropy)
	
	infogain = entropy_total - (((pxi+0.0)/total)*positive_entropy) - ((((total-pxi)+0.0)/total)*negative_entropy)
	return infogain		

def get_ones(data):
	one_values = 0
	for digit in data:		
		if digit == 1:
			one_values += 1

	return one_values

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable

# Get a particular column from total dataset
def get_column(data, i):
	return [row[i] for row in data]

# Split the "Class" data into two arrays, depending on other attribute values
def partition_data(data):
	data0 = []
	data1 = []
	index = 0
	row1 = data[0]
	row2 = data[1]

	for item in row1:
		if item == 0:
			data0.append(row2[index])
		elif item == 1:
			data1.append(row2[index])
		index += 1	

	return (data0,data1)

# Takes in the list of calculated entropies and returns the highest one
def split_data(data):
	best_value = 0.0
	best_value_index = 0
	i = 0
	for x in data:
		if x > best_value:
			best_value = x
			best_value_index = i
		i += 1
	return best_value,best_value_index

def count_values(data):
	data0 = 0
	data1 = 0
	for digit in data:
		if digit == 1:
			data1 += 1
		else:
			data0 += 1
	return(data0, data1)

def branch_data(data, column):
	left_branch = []
	right_branch = []

	for row in data:
		temp = row
		if row[column] == 0:
			left_branch.append(temp)
		else:
			right_branch.append(temp)
	return(left_branch, right_branch)

# Load data from a file
def read_data(filename):
	f = open(filename, 'r')
	p = re.compile(',')
	data = []
	header = f.readline().strip()
	varnames = p.split(header)
	namehash = {}
	for l in f:
		data.append([int(x) for x in p.split(l.strip())])
	return (data, varnames)

# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
	f = open(modelfile, 'w+')
	root.write(f, 0)

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):


	# Get class column length and count values of it
	class_length = get_column(data,len(varnames)-1)
	class_pos_neg = count_values(class_length)
	
	# Make leaf if only 0's
	if class_pos_neg[0]==len(class_length):
		return node.Leaf(varnames,0)
	# Make leaf if only 1's
	elif class_pos_neg[1]==len(class_length):
		return node.Leaf(varnames,1)

	else:
		# Get column of i and class column, then compute gain and return it
		returned_entropy = []
		for i in range(0,len(varnames)-1):
			columns = []
			columns.append(get_column(data,i))
			columns.append(get_column(data,len(data[0])-1))
			pos_and_neg = partition_data(columns)
			positive_values = pos_and_neg[1]
			class_values = columns[-1]
			returned_entropy.append(infogain(get_ones(positive_values),get_ones(columns[0]), get_ones(class_values), len(columns[0])))

		# Once we have a list of gains, get the highest one
		best_value = split_data(returned_entropy)
		# Branch the data into left and right, depending on higehst gain value in array
		branch = branch_data(data, best_value[1])

		# If the gain is exactly 0
		if best_value[0] == 0.0:
			if class_pos_neg[0] > class_pos_neg[1]:
				return node.Leaf(varnames,0)
			elif class_pos_neg[0] < class_pos_neg[1]:
				return node.Leaf(varnames,1)

		left = None
		right = None
		# There is no information gain progress
		if best_value[0] <= 0.0:

			# Get the columns of the branched data and count values in it
			check_branch_negatives = count_values(get_column(branch[0], best_value[1]))
			check_branch_positives = count_values(get_column(branch[1], best_value[1]))
			# Get length of both original branches
			length_neg = len(branch[0])
			length_pos = len(branch[1])

			# Make sure that neither the left, nor the right branch values are equal in length to the whole branch
			# That is, the tree needs to stop branching if one of the two branches is empty
			if (check_branch_negatives[0] == length_neg or check_branch_negatives[1] == length_neg):
				return node.Leaf(varnames, 1)
			elif (check_branch_negatives[0] < length_neg or check_branch_negatives[1] < length_neg):
				left = build_tree(branch[0], varnames)

			if (check_branch_positives[0] == length_pos or check_branch_positives[1] == length_pos):
				return node.Leaf(varnames, 1)
			elif (check_branch_positives[0] < length_pos or check_branch_positives[1] < length_pos):
				right = build_tree(branch[1], varnames)

			return node.Split(varnames, best_value[1], left, right)

		# The gain is higher than 0, everything is good and we can keep on branching 
		else:
			# We split the tree and recursively go through both new branches
			left = build_tree(branch[0], varnames)
			right = build_tree(branch[1], varnames)
		
			return node.Split(varnames, best_value[1], left, right)


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS,testS,modelS):
	global train
	global varnames
	global test
	global testvarnames
	global root
	(train, varnames) = read_data(trainS)
	(test, testvarnames) = read_data(testS)
	modelfile = modelS

	# build_tree is the main function you'll have to implement, along with
	# any helper functions needed.  It should return the root node of the
	# decision tree.
	root = build_tree(train, varnames)
	print_model(root, modelfile)
	
def runTest():
	correct = 0
	# The position of the class label is the last element in the list.
	yi = len(test[0]) - 1
	for x in test:
		# Classification is done recursively by the node class.
		# This should work as-is.
		pred = root.classify(x)
		if pred == x[yi]:
			correct += 1
	acc = float(correct)/len(test)
	return acc	
	
	
# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
	if (len(argv) != 3):
		print 'Usage: id3.py <train> <test> <model>'
		sys.exit(2)
	loadAndTrain(argv[0],argv[1],argv[2]) 
					
	acc = runTest()
	print "Accuracy: ",acc					  

if __name__ == "__main__":
	main(sys.argv[1:])