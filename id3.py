#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
# Node class for the decision tree
import node
import math

train=None
varnames=None
test=None
testvarnames=None
root=None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
	total_length = len(p)

	one_values = 0
	for digit in p:		
		if digit == 1:
			one_values += 1

	divided = (one_values+0.0)/total_length
	divided2 = ((total_length - one_values) + 0.0)/total_length
	if divided <= 0 or divided2 <= 0:
		return 0
	if divided == divided2:
		return 1
	entropy = -divided * math.log((divided),2) - ((divided2)* math.log((divided2),2))
	return entropy

def gain(data):
	column = data[-1]
	entropy_total = entropy(column)

	total_length = len(data[0])	
	pos_and_neg = partitiondata(data)

	negative_values = pos_and_neg[0]
	positive_values = pos_and_neg[1]

	if len(negative_values) != 0:
		negative_entropy = entropy(negative_values)
	else: 
		infogain = 0
		return infogain
	if len(positive_values) != 0:	
		positive_entropy = entropy(positive_values)
	else:
		infogain = 0
		return infogain

	infogain = entropy_total - (((len(positive_values)+0.0)/total_length)*positive_entropy) - (((len(negative_values)+0.0)/total_length)*negative_entropy)
	return infogain


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
	"total_entropy - (pos_sample/all_samples) * pos_sample_entropy - (neg_sample/all_samples) * neg_sample_entropy"
	# >>>> YOUR CODE GOES HERE <<<<
	# For now, always return "0":
	return 0;

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable

# Get a particular colum from total dataset
def get_column(data, i):
	return [row[i] for row in data]

# Split the "Class" data into two arrays, depending on other attribute values
def partitiondata(data):
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

def majority_value(data):
	data0 = 0
	data1 = 0
	for digit in data:
		if digit == 1:
			data1 = +1
		else:
			data0 += 1
	if data0 > data1:
		return 0
	else:
		return 1

def count_values(data):
	data0 = 0
	data1 = 0
	for digit in data:
		if digit == 1:
			data1 += 1
		else:
			data0 += 1
	return(data0, data1)

def deleteColumn(data, column):
	indices = []
	i = 0
	for i in range(len(data[0])):
		if i != column[1]:
			indices.append(i)
		i += 1	
	copyArray = [ [row[ci] for ci in indices] for row in data ]
	return copyArray

def deleteVar(varnames, column):
	print ("COLUMN", column[1])
	new_vars = varnames.pop(column[1])
	return varnames

def branch_data(data, column):
	left_branch = []
	right_branch = []
	i = 0

	for row in data:
		newRow = []
		temp = row
		if row[column] == 0:
			temp.pop(column)
			left_branch.append(temp)
		else:
			temp.pop(column)
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
	test = varnames[:]
	#return a guess by counting whether there are more 0's or 1's in the 
	best_guess = majority_value(get_column(data,len(data[0])-1))
	unambiguous = count_values(get_column(data,len(data[0])-1))

	# If the data is unambiguous, you can stop right away and make a leaf
	if (unambiguous[0] == len(data)) or (unambiguous[1] == len(data)):
		if unambiguous[0] == len(data):
			return node.Leaf(test, 0)
		else:
			return node.Leaf(test, 1)

	# If there are no more features left besides the "Class", you can stop as well and make a leaf
	elif len(varnames)<=1:
		return node.Leaf(test, best_guess)

	# Otherwise go through the data, calculate the information gain and branch
	else:
		total_length = len(data)
		returned_entropy = []
		i = 0
		for item in data:
			columns = []
			columns.append(get_column(data,i))
			columns.append(get_column(data,len(data[0])-1))
			returned_entropy.append(gain(columns))
			i += 1
			print len(varnames), i
			if i == len(varnames)-1:
				break

		# Once we have a list of gains, get the highest one
		best_value = split_data(returned_entropy)
		new_varnames = deleteVar(varnames, best_value)
		print best_value
		print new_varnames
		branch = branch_data(data, best_value[1])
		left = build_tree(branch[0], new_varnames)
		right = build_tree(branch[1], new_varnames)

		# Remove the column that we have selected so it won't show up again later
		#new_data =  deleteColumn(data, best_value)

		return node.Split(test, best_value[1], left, right)


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
