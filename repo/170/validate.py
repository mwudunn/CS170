import numpy as np
import os
import sys
import pickle
import networkx as nx
from networkx import algorithms as nxalgorithms
import matplotlib.pyplot as plt
import random as random
import math as math
from pathlib import Path

def read(num):
	"""Reads in the *.in file and returns a list representation"""
	file = open("./cs170_final_inputs/" + str(num) + ".in", "r" )
	adj_matrix = [line.split() for line in file.readlines()]
	num_nodes = int(adj_matrix[0][0])
	return num_nodes, np.array(adj_matrix[1:num_nodes+1])

def read_out(num):
	file = open("./out/" + str(num), "r" )
	paths = file.readlines()[0].split(";")
	for i in range(len(paths)):
		paths[i] = paths[i].split()
		for j in range(len(paths[i])):
			paths[i][j] = int(paths[i][j])
	return paths

def read_out_greedy(num):
	file = open("./out_greedy/" + str(num), "r" )
	paths = file.readlines()[0].split(";")
	for i in range(len(paths)):
		paths[i] = paths[i].split()
		for j in range(len(paths[i])):
			paths[i][j] = int(paths[i][j])
	return paths

def read_out_greedy_score(num):
	file = open("./out_greedy/" + str(num), "r" )
	score = int(file.readlines()[2])
	
	return score

def create_graph(num, adjancency_matrix):
	"""Returns a networkx graph"""
	G = nx.DiGraph()
	nodes = adjancency_matrix.diagonal()
	for i in range(num): #add nodes into graph
		G.add_node(i, weight=int(nodes[i]))
	for j in range(adjancency_matrix.shape[0]): #add edges
		for i in range(adjancency_matrix.shape[1]):
			if i == j:
				continue
			if int(adjancency_matrix[j,i]) == 1:
				G.add_edge(j, i)
	return G

def score_partitions(G,paths):
	total = 0
	flatten = sum(paths, [])
	#print(flatten)
	for path in paths:
		for i in range(0,len(path)-1): # confirm that adjacent pairs have an edge
			if G.has_edge(path[i], path[i+1]) != True:
				# print("path from " + str(path[i]) + " " +str(path[i+1]))
				return -1
		total += score(G,path)
	for node in G.nodes():
		if flatten.count(node) != 1:
			print("error")
			return -1
	return total

# Gives the score for a single path
def score(G, path):
	weights = nx.get_node_attributes(G, "weight")
	total = 0
	for node in path:
		total += weights[node]
	return total * len(path)

def check(i):
	a, b = read(int(i))
	g = create_graph(a, b)
	#out_check = read_out(i)
	out_two = read_out_greedy(i)
	#return score_partitions(g, out_check), score_partitions(g, out_two)
	return score_partitions(g, out_two)

fout = open("./scores.txt", "w")
for i in range(1, 601):
	# f1 = Path("./out/"+ str(i))
	# f2 = Path("./out_greedy/"+ str(i))
	# if f1.is_file() and f2.is_file():
	# 	print("comparing: " + str(i))
	# 	print(check(i))
	# else:
	# 	print("couldn't find: " + str(i))
	
	fout.write(str(i) + " " +  str(read_out_greedy_score(i)) + "\n")







