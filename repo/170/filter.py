import numpy as np  
import os
import sys
import pickle
import networkx as nx
import matplotlib.pyplot as plt

def read(num):
	"""Reads in the *.in file and returns a list representation"""
	file = open("./cs170_final_inputs/" + str(num) + ".in", "r" )
	adj_matrix = [line.split() for line in file.readlines()]
	num_nodes = int(adj_matrix[0][0])
	return num_nodes, np.array(adj_matrix[1:num_nodes+1])

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

def hamilton(G, num, path=False):
	paths = [p for p in nx.all_simple_paths(G, source=0, target=0) if len(p) == num + 1]
	if path:
		return paths
	else:
		return len(paths) == 0



def find_small_nodes(cut_off=30, tolerance = 3):
	small_nodes = []
	for i in range(1, 601):
		num, matrix = read(i)
		if i%10 == 0:
			print(i)
		if num <= 25:
			g = create_graph(num, matrix)
			if nx.number_of_edges(g) <= 3 * num:
				small_nodes.append(int(i))
	pickle.dump(np.array(small_nodes), open("./small_nodes", "wb"))
	return np.array(small_nodes)

def find_sparse(tolerance = 2):
	sparse = []
	for i in range(1, 601):
		num, matrix = read(i)
		if i%10 == 0:
			print(i)
		g = create_graph(num, matrix)
		if nx.number_of_edges(g) <= tolerance * num:
			sparse.append(int(i))
	pickle.dump(np.array(sparse), open("./sparse", "wb"))
	return np.array(sparse)

def is_hamilton(small_nodes):
	hamiltonian = []
	for i in small_nodes:
		num, matrix = read(i)
		print(i,num)
		g = create_graph(num, matrix)
		if hamilton(g, num):
			print("yes")
			print("")
			hamiltonian.append(i)
	pickle.dump(np.array(hamiltonian), open("./hamiltonian", "wb"))
	return np.array(hamiltonian)

find_small_nodes()
small_nodes = pickle.load(open("small_nodes", "rb"))
find_sparse()
sparse = pickle.load(open("sparse", "rb"))
nodes = np.concatenate(small_nodes, sparse)
nodes = set()
for entry in small_nodes:
	nodes.add(entry)
for entry in sparse:
	nodes.add(entry)
nodes = list(nodes)
print(nodes)
is_hamilton(nodes)
