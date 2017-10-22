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

# given a partition (a solution) of G, check its validity and tally the score. return -1 if invalid paths
def score_partitions(G,paths):
    total = 0
    visited = []
    for path in paths:
        for each in path:
            visited.append(each)
        for i in range(0,len(path)-1): # confirm that adjacent pairs have an edge
            if G.has_edge(path[i], path[i+1]) != True:
                # print("path from " + str(path[i]) + " " +str(path[i+1]))
                return -1
        total += score(G,path)
    if len(visited) != nx.number_of_nodes(G):
        print("not all nodes used")
        return -1
    return total

def dosmething(input, input2):
    x = input
    x = set(input)
    sort(x)
    for i in range(x):
        input[i] = 5
        
    return x

# Gives the score for a single path
def score(G, path):
    weights = nx.get_node_attributes(G, "weight")
    total = 0
    for node in path:
        total += weights[node]
    return total * len(path)

def randomized_one(g):
    G = g.copy()
    allNodes = set(G.nodes())
    allTeams = []
    while len(allNodes) > 0:
        team = []
        node = random.sample(allNodes, 1)[0]
        team.append(node)
        nodeToDegree = {}
        neighbors = validNeighbors(team, G.neighbors(node))
        nodeToDegree[node] = float(numOfValidNeighbors(team, G.predecessors(node)) * numOfValidNeighbors(team, G.neighbors(node)))
        while len(neighbors) > 0:
            node = random.sample(neighbors, 1)[0]
            neighbors = validNeighbors(team, G.neighbors(node))
            team.append(node)
            nodeToDegree[node] = float(numOfValidNeighbors(team, G.predecessors(node)) * numOfValidNeighbors(team, G.neighbors(node)))
        team = shortenPath(G, nodeToDegree, team)
        if len(team) == 0:
            continue
        allTeams.append(team)
        for node in team:
            G.remove_node(node)
            allNodes.remove(node)
    return allTeams

# Shortens a path we have found so far depending on the following factors:
# 1. The longer a path is, the less we want to sever the path. Thus
# say that we start @ node A, if the length of the path is the same size as A's
# connected component, then we do not sever it, however if it is much shorter,
# then we have a much higher chance of severing it
# 2. The spot @ which we sever depends on the degree of each vertex, which is
# passed in through NODETODEGREE
def shortenPath(G, nodeToDegree, team):
    teamSize = len(team)
    if teamSize == 1:
        return team
    severingChance = 1.0 - float(teamSize) / float(G.number_of_nodes())
    if random.random() >= severingChance:
        return team
    degreeSum = float(sum(nodeToDegree.values()))
    prev = 0.0
    pick = random.random()
    for i in range(len(team)):
        prev = prev + float(nodeToDegree[team[i]]) / degreeSum
        if prev >= pick:
            return team[:i]


# Returns True if there are vertices that have not been explored in NEIGHBORS
def checkNeighbors(nodesExplored, neighbors):
    return len(validNeighbors(nodesExplored, neighbors)) > 0

# Returns the number of subset of neighbors that have not been explored yet
# This function will be used in calculating the degree of each vertex,
# aka in-degree * out-degree. If this causes a dramatic slowdown we can take
# it out
def numOfValidNeighbors(nodesExplored, neighbors):
    return len(validNeighbors(nodesExplored, neighbors)) + 1

def validNeighbors(nodesExplored, neighbors):
    valid = set()
    for neighbor in neighbors:
        if not neighbor in nodesExplored:
            valid.add(neighbor)
    return valid

def randomized_two(g):
    G = g.copy()
    allNodes = set(G.nodes())
    allTeams = []
    def sample(nodesToChooseFrom):
        node = random.sample(nodesToChooseFrom, 1)[0]
        in_degree = len(G.predecessors(node))
        out_degree = len(G.successors(node))
        if in_degree * out_degree > 0:
            chance = math.exp(-in_degree * out_degree)
            if random.random() < chance:
                return None, None
        nodesToChooseFrom.remove(node)
        neighbors = G.neighbors(node)
        G.remove_node(node)
        return node, neighbors
    while len(allNodes) > 0:
        team = []
        node, neighbors = sample(allNodes)
        if node == None:
            continue
        team.append(node)
        while neighbors != None and len(neighbors) > 0:
            node, neighbors = sample(neighbors)
            team.append(node)
        allTeams.append(team)
    return allTeams

def randomized_three(g):
    G = g.copy()
    allNodes = set(G.nodes())
    allTeams = []
    def sample(nodesToChooseFrom):
        node = random.sample(nodesToChooseFrom, 1)[0]
        allNodes.remove(node)
        neighbors = G.neighbors(node)
        G.remove_node(node)
        return node, neighbors
    while len(allNodes) > 0:
        team = []
        node, neighbors = sample(allNodes)
        if node == None:
            continue
        team.append(node)
        while neighbors != None and len(neighbors) > 0:
            node, neighbors = sample(neighbors)
            team.append(node)
        allTeams.append(team)
    return allTeams

def compute_path_from_node_one(node, subgraph, weights, prob):
    successors = subgraph.successors(node)
    predecessors = subgraph.predecessors(node)

    path = [node]
    seenSoFar = set()
    seenSoFar.add(node)
    while len(successors) > 0:
        maxNeighbor = None
        bestRatio = -1
        for each in successors:
            if not each in seenSoFar:
                inDegree = subgraph.in_degree(each)
                outDegree = subgraph.out_degree(each) + 1 #ensure non-zero out-degree
                degreeRatio = inDegree / outDegree #Only works in python 3+
                if degreeRatio < bestRatio or bestRatio == -1:
                    if random.random() < prob:
                        continue;
                    maxNeighbor = each
                    bestRatio = degreeRatio
        if maxNeighbor == None:
            break;
        successors = subgraph.successors(maxNeighbor)
        path.append(maxNeighbor)
        seenSoFar.add(maxNeighbor)


    while len(predecessors) > 0:
        maxNeighbor = None
        bestRatio = -1
        for each in predecessors:
            if not each in seenSoFar:
                if random.random() < prob:
                    continue;
                inDegree = subgraph.in_degree(each)
                outDegree = subgraph.out_degree(each) + 1 #ensure non-zero out-degree
                degreeRatio = inDegree / outDegree #Only works in python 3+
                if degreeRatio < bestRatio or bestRatio == -1:
                    maxNeighbor = each
                    bestRatio = degreeRatio
        if maxNeighbor == None:
            break;
        predecessors = subgraph.predecessors(maxNeighbor)
        path.insert(0, maxNeighbor)
        seenSoFar.add(maxNeighbor)

    return path

def greedy_one(G, prob):
    weights = nx.get_node_attributes(G, "weight")
    G = nxalgorithms.weakly_connected_component_subgraphs(G, True)
    allTeams = []
    allGraphs = [g for g in G]

    def path_in_component(sub): #sub is a WCC subcomponent
        allNodes = sub.nodes()
        if len(allNodes) == 0:
            return
        bestValue = -1
        bestPath = []

        for node in allNodes:
            path = compute_path_from_node_one(node, sub, weights, prob)
            if score(sub, path) > bestValue:
                bestValue = score(sub, path)
                bestPath = path
            if len(path) == len(allNodes):
                break

        allTeams.append(bestPath)
        sub.remove_nodes_from(bestPath)
        newSubgraphs = nxalgorithms.weakly_connected_component_subgraphs(sub, True)
        for sub in newSubgraphs:
            allGraphs.append(sub)

    while len(allGraphs) > 0:
        sub = allGraphs[0]
        allGraphs = allGraphs[1:]
        path_in_component(sub)
    return allTeams 


def compute_path_from_node_three(node, subgraph, weights):
    successors = subgraph.successors(node)
    predecessors = subgraph.predecessors(node)

    path = [node]
    seenSoFar = set()
    seenSoFar.add(node)
    while len(successors) > 0:
        maxNeighbor = None
        best_score = -1
        for each in successors:
            if not each in seenSoFar:
                score = weights[each]
                if score > best_score or best_score == -1:
                    maxNeighbor = each
                    best_score = score
        if maxNeighbor == None:
            break
        successors = subgraph.successors(maxNeighbor)
        path.append(maxNeighbor)
        seenSoFar.add(maxNeighbor)

    while len(predecessors) > 0:
        maxNeighbor = None
        best_score = -1
        for each in predecessors:
            if not each in seenSoFar:
                score = weights[each]
                if score > best_score or best_score == -1:
                    maxNeighbor = each
                    best_score = score
        if maxNeighbor == None:
            break
        predecessors = subgraph.predecessors(maxNeighbor)
        path.insert(0, maxNeighbor)
        seenSoFar.add(maxNeighbor)

    return path

longestGreedyPath = {}
longestGreedyPathLength = -1

def greedy_three(G):
    GG = G
    g = G.copy()
    weights = nx.get_node_attributes(G, "weight")
    G = nxalgorithms.weakly_connected_component_subgraphs(G, True)
    allTeams = []
    allGraphs = [g for g in G]
    longestPathLength = 0

    def path_in_component(sub): #sub is a WCC subcomponent
        allNodes = sub.nodes()
        if len(allNodes) == 0:
            return
        bestValue = -1
        bestPath = []
        longestPathLength = 0

        for node in allNodes:
            path = compute_path_from_node_three(node, sub, weights)
            if score(sub, path) > bestValue:
                bestValue = score(sub, path)
                bestPath = path

        if len(bestPath) > longestPathLength:
            longestPathLength = len(bestPath)
        allTeams.append(bestPath)
        sub.remove_nodes_from(bestPath)
        newSubgraphs = nxalgorithms.weakly_connected_component_subgraphs(sub, True)
        for sub in newSubgraphs:
            allGraphs.append(sub)
        return longestPathLength

    while len(allGraphs) > 0:
        sub = allGraphs[0]
        allGraphs = allGraphs[1:]
        length = path_in_component(sub)
        if length > longestPathLength:
            longestPathLength = length

    longestGreedyPath[GG] = longestPathLength
    return allTeams

def compute_path_from_node_random(node, subgraph, weights, longestPathLength):
    successors = subgraph.successors(node)
    predecessors = subgraph.predecessors(node)
    chanceOfDeviation = 1.0 - 0.5 ** (1.0 / float(longestPathLength))

    path = [node]
    seenSoFar = set()
    seenSoFar.add(node)
    while len(successors) > 0:
        maxNeighbor = None
        best_score = -1
        for each in successors:
            if not each in seenSoFar:
                score = weights[each]
                if score > best_score or best_score == -1:
                    maxNeighbor = each
                    best_score = score
        if random.random() < chanceOfDeviation:
            choices = [node for node in successors if not node in seenSoFar]
            if len(choices) == 0:
                maxNeighbor = None
            else:
                maxNeighbor = random.choice(choices)
        if maxNeighbor == None:
            break
        successors = subgraph.successors(maxNeighbor)
        path.append(maxNeighbor)
        seenSoFar.add(maxNeighbor)

    while len(predecessors) > 0:
        maxNeighbor = None
        best_score = -1
        for each in predecessors:
            if not each in seenSoFar:
                score = weights[each]
                if score > best_score or best_score == -1:
                    maxNeighbor = each
                    best_score = score
        if random.random() < chanceOfDeviation:
            choices = [node for node in predecessors if not node in seenSoFar]
            if len(choices) == 0:
                maxNeighbor = None
            else:
                maxNeighbor = random.choice(choices)
        if maxNeighbor == None:
            break
        predecessors = subgraph.predecessors(maxNeighbor)
        path.insert(0, maxNeighbor)
        seenSoFar.add(maxNeighbor)

    return path

def greedy_random(G):
    GG = G
    g = G.copy()
    weights = nx.get_node_attributes(G, "weight")
    G = nxalgorithms.weakly_connected_component_subgraphs(G, True)
    allTeams = []
    allGraphs = [g for g in G]
    longestPathLength = longestGreedyPath[GG]

    def path_in_component(sub): #sub is a WCC subcomponent
        allNodes = sub.nodes()
        if len(allNodes) == 0:
            return
        bestValue = -1
        bestPath = []

        for node in allNodes:
            path = compute_path_from_node_random(node, sub, weights, longestPathLength)
            if score(sub, path) > bestValue:
                bestValue = score(sub, path)
                bestPath = path

        allTeams.append(bestPath)
        sub.remove_nodes_from(bestPath)
        newSubgraphs = nxalgorithms.weakly_connected_component_subgraphs(sub, True)
        for sub in newSubgraphs:
            allGraphs.append(sub)

    while len(allGraphs) > 0:
        sub = allGraphs[0]
        allGraphs = allGraphs[1:]
        path_in_component(sub)

    return allTeams

#Checks for hamiltonian cycles
def hamilton_check(g, b):
    diagonal = [int(i) for i in b.diagonal()]
    if len(diagonal) > 25:
        print("Too  Long")
        return False, False, -1
    if nx.number_of_edges(g) > .3 * ( nx.number_of_nodes(g)  ** 2):
        print("Too  Bushy")
        return False, False, -1
    total = sum(diagonal) * len(diagonal)
    for i in range(len(diagonal)):
        for j in range(len(diagonal)):
            if i == j:
                continue
            all_paths = [gph for gph in nx.all_simple_paths(g, i, j) if len(gph) == len(diagonal)]
            if all_paths != []:
                print("Done Hamiltonian")
                return True, [all_paths[0]], -1
    
    return False, False, -1

#Attempts to concatenate paths together to generate better results
def concatenation(G, teams=[]):
    subs = nxalgorithms.weakly_connected_component_subgraphs(G, True)

    allGraphs = [g for g in subs]
    if len(teams) < 2:
        return teams
    new_teams = list(teams)
    first_team = list(teams[0])
    second_team = list(teams[1])
    first_team.reverse()
    best = 501
    sec_count = 0
    current_path = []
    best_path = []
    best_second_path = []
    remaining_second_path = list(second_team)
    best_first_length = 0
    for sec_node in second_team:
        sec_count += 1
        fir_count = 0
        current_path.append(sec_node)
        new_path_first = []
        already_found = 0
        for fir_node in first_team:
            fir_count += 1
            new_path_first.append(fir_node)
            if already_found == 1:
                continue
            first_successors = allGraphs[0].successors(fir_node)
            if sec_node in first_successors:
               
                if sec_count + fir_count < best:
                    print("Concatenation possible")
                    full_first_path = first_team[fir_count - 1:]
                    full_first_path.reverse()
                    first_team[fir_count:]
                    rest_second_path = second_team[sec_count - 1:]
                    best_path = full_first_path + rest_second_path
                    already_found = 1
                    best = sec_count + fir_count
                    best_first_length = len(first_team) - fir_count
                    remaining_second_path = current_path[:-1]
        if already_found == 1:
            best_second_path = new_path_first
    if best_path:
        new_teams[0] = best_path
        first_team.reverse()
        new_second = first_team[best_first_length + 1:]
        recursive_concatenation = concatenation(G, [new_second, remaining_second_path])
        recursive_concatenation[0].reverse()
        for each in recursive_concatenation:
            new_teams.append(each)
    return new_teams

def concatenator(G, teams=[]):

    weights = nx.get_node_attributes(G, "weight")
    g = nxalgorithms.weakly_connected_component_subgraphs(G, True)
    allTeams = list(teams)
    allGraphs = [g for g in G]
    nextList = []
    if len(allTeams) >= 2:
        newList = concatenation(G, [allTeams[0], allTeams[1]])
        for i in range(len(newList)):
            if newList[i]:
                nextList.append(newList[i])
        for j in range(len(allTeams) - 2):
            nextList.append(allTeams[j + 2])
    else:
        nextList = allTeams
    print(nextList)
    return nextList





    def path_in_component(sub): #sub is a WCC subcomponent
        allNodes = sub.nodes()
        if len(allNodes) == 0:
            return
        bestValue = -1
        bestPath = []

        for node in allNodes:
            path = compute_path_from_node_one(node, sub, weights, prob)
            if score(sub, path) > bestValue:
                bestValue = score(sub, path)
                bestPath = path
            if len(path) == len(allNodes):
                break

        allTeams.append(bestPath)
        sub.remove_nodes_from(bestPath)
        newSubgraphs = nxalgorithms.weakly_connected_component_subgraphs(sub, True)
        for sub in newSubgraphs:
            allGraphs.append(sub)

    while len(allGraphs) > 0:
        sub = allGraphs[0]
        copyOfSub = list(sub)
        allGraphs = allGraphs[1:]
        path_in_component(sub)
    return allTeams 
def concatenate_teams(G, teams=[]):
    computation = concatenator(G, teams)
    return computation, score_partitions(G, computation)

def generate_solution(f, g):
    # nx.draw_networkx(g)
    # plt.show()
    computation = f(g)
    return computation, score_partitions(g, computation)

def generate_greedy_random_solution(G, prob):
    computation = greedy_one(G, prob)
    return computation, score_partitions(G, computation)

def FileCheck(filename):
    try:
      f = open(filename, "r")
      return f
    except IOError:
      print("Error: " + filename + " does not appear to exist.")
      return False

def read_sol(num):
    file = FileCheck("./out/" + str(num))
    if file != False:
        solution_string = file.readline()
        solution_list = solution_string.split(";")
        teams = [x.split() for x in solution_list]
        for path in teams:
            for i in range(len(path)):
                path[i] = int(path[i])
        FLAG = int(file.readline())
        score = int(file.readline())
        file.close()
        return teams, score
    else:
        print(str(num) + " is not in the saved solutions")
        raise IOError
def instance(inpt, iterations, iterations_greedy_one, prob): #[start, finish]
    FLAG = 0
    a, b = read(inpt)
    G = create_graph(a, b)
    teams, score = read_sol(inpt)
    best = [teams, score]
#     teams, score = concatenate_teams(G, teams)
#     print(score, " ", best[1])
#     if score > best[1]:
#         FLAG = 3
#         return teams, score, FLAG

    # best = [teams, score]
    # if len(teams) == 1:
    #     print("best solution already found")
    #     return best[0], best[1], FLAG
    teams, score = generate_greedy_random_solution(G, 0)
    # best = [teams, score]
    # if len(teams) == 1:
    #     print("best solution already found")
    #     print(score)
    #     return best[0], best[1], FLAG
    # if len(teams) == 1:
    #     print("greedy one hamiltonian")
    #     return best[0], best[1], FLAG
    teams, score = generate_solution(greedy_three, G)
    # if best[1] < score:
    #     best[1] = score
    #     best[0] = teams
    #     FLAG = 1
    # if len(teams) == 1:
    #     print("best solution already found")
    #     return best[0], best[1], FLAG
    # for _ in range(iterations_greedy_one): #Randomizes the new algorithm
    #     if _ % 10 == 0:
    #         print("Iteration: ", _)
    #         print("Current score: ", best[1])
    #     teams, score = generate_greedy_random_solution(G, prob) 
    #     if best[1] < score:
    #         best[1] = score
    #         best[0] = teams
    #         FLAG = 2
    #         if len(teams) == 1:
    #             return best[0], best[1], FLAG
    # found_better = False
    # for _ in range(iterations):
    #     if _ % 10 == 0:
    #         print("Iteration: ", _)
    #         print("Current score: ", best[1])
    #     teams, score = generate_solution(greedy_random, G)
    #     if best[1] < score:
    #         best[1] = score
    #         best[0] = teams
    #         FLAG = 2
    #         found_better = True
    #         if len(teams) == 1:
    #             print("found better")
    #             return best[0], best[1], FLAG
    # if found_better == True:
    #     print("found better")
    return best[0], best[1], FLAG

sys.setrecursionlimit(1000)

def main(start, finish, iterations=5, iterations_greedy=5, prob=0.2): #prob = probability of passing on an "optimal" node for randomized greedy_one
    
    for i in range(start, finish+1):
        print("current instance: ", i)
        a, b = read(i)
        G = create_graph(a, b)
        # nx.draw_networkx(G)
        # plt.show()
        check, teams, FLAG = hamilton_check(G, b)
        if check:   
            scr = score(G, teams[0])
        else:
            teams, scr, FLAG = instance(i, iterations, iterations_greedy, prob)
        if (FLAG == 3):
            print("this instance's path: ", teams)
            f = open("./Better/" + str(i), "w")
        else:
            print(teams)
            f = open("./Better/" + str(9999), "w")
        write_out = ""
        for i in range(len(teams)):
            if i != 0:
                write_out += ";"
            for j in range(len(teams[i])):
                write_out += " " + str(teams[i][j])
        write_out = write_out[1:]
        write_out += "\n" + str(FLAG) + "\n" + str(scr)
        f.write(write_out)
    return teams
test = main(int(sys.argv[1]), int(sys.argv[2]))
