import numpy as np
import os
import sys
import pickle
import networkx as nx
from networkx import algorithms as nxalgorithms
import matplotlib.pyplot as plt
import random as random
import math as math

def read(num):
	"""Reads in the *.in file and returns a list representation"""
	file = open("./cs170_final_inputs/" + str(num) + ".in", "r" )
	adj_matrix = [line.split() for line in file.readlines()]
	num_nodes = int(adj_matrix[0][0])
	return num_nodes, np.array(adj_matrix[1:num_nodes+1])

def read_sol(num):

	file = FileCheck("./out/" + str(num))
	if file != False:
		solution = file.readline()
		file.close()
		return solution
	else:
		return False

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
				G.add_edge(i, j)
	return G

def score_partitions(G,paths):
	total = 0
	flatten = sum(paths, [])
	print(flatten)
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

def FileCheck(filename):
    try:
      f = open(filename, "r")
      return f
    except IOError:
      print("Error: " + filename + " does not appear to exist.")
      return False

def main(i, j):
	f = open("answer.txt", "w")
	for i in range(i, j + 1):
		out = read_sol(i)
		if out != False:
			f.write(out)
		else:
			print(str(i) + " is not there")
			f.write("\n")
	f.close()
	


main(1, 600)
