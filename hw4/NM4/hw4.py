# Please enter here the netids of all members of your group (yourself included.)
authors = ['dh649', 'aj545', 'dz336']

# Which version of python are you using? python 2 or python 3? 
python_version = "python 3.6.2."

# Important: You are NOT allowed to modify the method signatures (i.e. the arguments and return types each function takes).

# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. 
# Assume that nodes are represented by indices between 0 and number_of_nodes - 1
import numpy as np
import random
import matplotlib.pyplot as plt


class DirectedGraph:
    
    def __init__(self,number_of_nodes):
        self.__number_of_nodes = number_of_nodes
        self.adjacency_matrix = np.zeros( (number_of_nodes, number_of_nodes) )

    def add_edge(self, origin_node, destination_node):
        self.adjacency_matrix[
            origin_node,
            destination_node
        ] = 1

    def out_degree(self, origin_node):
        return(
            len(self.edges_from(origin_node=origin_node))
        )


    
    def edges_from(self, origin_node):
        ''' This method should return a list of all the nodes u such that the edge (origin_node,u) is
        part of the graph.'''
        return(
            list(np.where(self.adjacency_matrix[origin_node, :] == 1)[0])
        )
    
    def check_edge(self, origin_node, destination_node):
        ''' This method should return true is there is an edge between origin_node and destination_node
        and destination_node, and false otherwise'''
        return(
            self.adjacency_matrix[origin_node, destination_node] == 1
        )

    def number_of_nodes(self):
        ''' This method should return the number of nodes in the graph'''
        return(
            self.__number_of_nodes
        )

    def get_nodes(self):
        return(
            list(range(self.number_of_nodes()))
        )

    def edges_to(self, destination_node):
        return (
            list(np.where(self.adjacency_matrix[:, destination_node] == 1)[0])
        )



def scaled_page_rank(graph, num_iter, eps = 1/7.0):
    ''' This method, given a directed graph, should run the epsilon-scaled page-rank
    algorithm for num-iter iterations and return a mapping (dictionary) between a node and its weight. 
    In the case of 0 iterations, all nodes should have weight 1/number_of_nodes'''

    # Init page rank values
    page_ranks = dict(zip(
        graph.get_nodes(),
        np.full((1, graph.number_of_nodes()), 1 / graph.number_of_nodes())[0]
    ))
    if num_iter == 0:
        return(page_ranks)


    # Calculate progress breakpoints
    breakpoints = [int(i) for i in np.round(np.linspace(1, num_iter, num=10), 0)]

    # Begin iterating
    for iter in range(num_iter):
        # debugging help...
        if iter in breakpoints:
            print("{}% completed...".format(
                breakpoints.index(iter) * 10
            ))

        # Init current iteration page ranks
        tmp_page_ranks = dict()

        for node in page_ranks.keys():
            # Init and calculate dangling nodes value
            dangling_value = 0
            if len(graph.edges_from(origin_node=node))==0:
                dangling_value = page_ranks[node]

            # Run current page rank calculation
            tmp_page_ranks[node] = (eps/graph.number_of_nodes()) + \
                               (1-eps) * \
                               (
                                   np.sum(
                                       [
                                           page_ranks[other_node] / graph.out_degree(other_node)
                                           for other_node in graph.edges_to(node)
                                       ]
                                   ) + dangling_value
                               )

            # alternative implementation (found on Priceton's site)
            # tmp_page_ranks[node] = (1 - eps) + \
            #                        (eps) * \
            #                        (
            #                            np.sum(
            #                                [
            #                                    page_ranks[other_node] / graph.out_degree(other_node)
            #                                    for other_node in graph.edges_to(node)
            #                                ]
            #                            ) +
            #                            dangling_value
            #                        )
        page_ranks = tmp_page_ranks

    # Check output
    if round(sum(page_ranks.values()), 2) != 1.0:
        raise ValueError("Sum of page rank is not close enough to 1.")

    return(page_ranks)



def graph_15_1_left():
    ''' This method, should construct and return a DirectedGraph encoding the left example in fig 15.1
    Use the following indexes: A:0, B:1, C:2, Z:3 '''
    g = DirectedGraph(4)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(0, 3)
    return(g)

def graph_15_1_right():
    ''' This method, should construct and return a DirectedGraph encoding the right example in fig 15.1
    Use the following indexes: A:0, B:1, C:2, Z1:3, Z2:4'''
    g = DirectedGraph(5)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(0, 3)
    g.add_edge(0, 4)
    g.add_edge(3, 4)
    g.add_edge(4, 3)
    return (g)

def graph_15_2():
    ''' This method, should construct and return a DirectedGraph encoding example 15.2
        Use the following indexes: A:0, B:1, C:2, A':3, B':4, C':5'''
    g = DirectedGraph(6)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    g.add_edge(5, 3)
    return (g)

def extra_graph_1():
    ''' This method, should construct and return a DirectedGraph of your choice with at least 10 nodes'''    
    g = DirectedGraph(12)
    g.add_edge(0, 1)
    g.add_edge(1, 0)
    g.add_edge(0, 2)
    g.add_edge(2, 0)
    g.add_edge(0, 3)
    g.add_edge(3, 0)

    g.add_edge(3, 4)
    g.add_edge(3, 5)
    g.add_edge(3, 6)
    g.add_edge(3, 7)

    g.add_edge(3, 8)
    g.add_edge(3, 9)
    g.add_edge(3, 10)
    g.add_edge(3, 11)

    g.add_edge(8, 0)
    g.add_edge(9, 0)
    g.add_edge(10, 0)
    g.add_edge(11, 0)

    return(g)

# This dictionary should contain the expected weights for each node when running the scaled page rank on the extra_graph_1 output
# with epsilon = 0.07 and num_iter = 20.
# extra_graph_1_weights = {1 : 0, 2: 0, 3 : 0, 4: 0, 5 : 0, 6: 0, 7 : 0, 8: 0, 9 : 0}
extra_graph_1_weights = scaled_page_rank(graph=extra_graph_1(), num_iter = 50)


def extra_graph_2():
    ''' This method, should construct and return a DirectedGraph of your choice with at least 10 nodes'''    
    g = DirectedGraph(11)

    g.add_edge(0, 2)
    g.add_edge(2, 0)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 6)
    g.add_edge(6, 4)
    g.add_edge(6, 3)
    g.add_edge(5, 6)

    g.add_edge(3, 7)
    g.add_edge(3, 8)
    g.add_edge(3, 9)

    g.add_edge(7, 10)
    g.add_edge(8, 10)
    g.add_edge(9, 10)

    return(g)

# This dictionary should contain the expected weights for each node when running the scaled page rank on the extra_graph_2 output
# with epsilon = 0.07 and num_iter = 20.
# extra_graph_2_weights = {1 : 0, 2: 0, 3 : 0, 4: 0, 5 : 0, 6: 0, 7 : 0, 8: 0, 9 : 0}

extra_graph_2_weights = scaled_page_rank(graph=extra_graph_2(), num_iter = 50)



def facebook_graph(filename = "facebook_combined.txt"):
    ''' This method should return a DIRECTED version of the facebook graph as an instance of the DirectedGraph class.
    In particular, if u and v are friends, there should be an edge between u and v and an edge between v and u.'''
    with open(filename, mode="r") as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]

    g = DirectedGraph(4039)
    for edge in content:
        g.add_edge(origin_node=int(edge[0]), destination_node=int(edge[1]))
        g.add_edge(origin_node=int(edge[1]), destination_node=int(edge[0]))

    return(g)


# The code necessary for your measurements for question 8b should go in this function.
# Please COMMENT THE LAST LINE OUT WHEN YOU SUBMIT (as it will be graded by hand and we do not want it to interfere
# with the automatic grader).
def question8b():

    # Load Facebook graph
    filepath = "/Users/davidhachuel/Google Drive/Cornell Tech/FALL_2017/CS_5854_NETWORKS_&_MARKETS/HW4/data/facebook_combined.txt"
    FB_g = facebook_graph(filename=filepath)

    # Run PR for 20 iterations
    pr = scaled_page_rank(graph=FB_g, num_iter=20)

    return(pr)







########################################################################################################################
## UNIT TESTS
########################################################################################################################
# testGraph = DirectedGraph(5)
#
# assert testGraph.number_of_nodes() == 5
#
# assert testGraph.check_edge(0,1) == False
#
# testGraph.add_edge(0,1)
#
# assert testGraph.check_edge(0,1) == True
#
# weights = scaled_page_rank(testGraph,0)
#
# assert weights[2] == 1/5.0
#
# assert graph_15_1_left().number_of_nodes() == 4
########################################################################################################################











########################################################################################################################
## TESTING QUESTION 8
########################################################################################################################
## Run PR on Facebook graph
## ------------------------
# import networkx as nx
# import matplotlib.pyplot as plt
# pr = question8b()
# pr_sorted = sorted(pr.items(), key=lambda x:x[1])
# high_20_fb_pr_idx = [item[0] for item in pr_sorted[-20:]]
# low_20_fb_pr_idx = [item[0] for item in pr_sorted[0:20]]
#
# # Load Facebook data for analysis
# # -------------------------------
# filepath = "~/data/facebook_combined.txt"
# FB_g = nx.DiGraph(facebook_graph(filename=filepath).adjacency_matrix)
#
# # Basic Box Plot of PR
# # --------------------
# from sklearn.preprocessing import normalize
# close = nx.closeness_centrality(G=FB_g)
# degree = nx.degree(G=FB_g)
# plt.boxplot([
#     np.log(normalize(np.array(list(pr.values())).reshape(-1,1), axis=0)),
#     np.log(normalize(np.array(list(close.values())).reshape(-1, 1), axis=0)),
#     np.log(normalize(np.array(list(degree.values())).reshape(-1, 1), axis=0))
# ])
# plt.suptitle("Page Rank VS Closeness VS Degree Centrality Distribution in Facebook Graph")
# plt.ylabel("Log of Normalize Scores")
# plt.xticks([1, 2, 3],['Page Rank', 'Closeness', 'Degree'], rotation=45, fontsize=8)
# plt.show()


## Trend plots
## -----------
# nodes = list(pr.keys())
# pagerank = [pr[node] for node in nodes]
# avg_neighbor_degree = list(nx.average_neighbor_degree(G=FB_g, nodes=nodes).values())
# degree = list(nx.degree(G=FB_g, nbunch=nodes).values())
# avg_neighbor_page_rank = [np.mean([pr[neighbor] for neighbor in nx.neighbors(G=FB_g, n=node)]) for node in nodes]
# plt.figure(1)
# plt.subplot(121)
# plt.scatter(avg_neighbor_degree, pagerank, c=pagerank)
# plt.xlabel("Average Neighbor Degree")
# plt.ylabel("Page Rank")
# plt.title("")
# plt.subplot(122)
# plt.scatter(degree, pagerank, c=pagerank)
# plt.xlabel("Node Degree")
# plt.ylabel("Page Rank")
# plt.title("")
# plt.show()
########################################################################################################################











########################################################################################################################
## TESTING QUESTION 7
########################################################################################################################
# test_graphs = {
#     "graph_15_1_left":graph_15_1_left(),
#     "graph_15_1_right":graph_15_1_right(),
#     "graph_15_2":graph_15_2(),
#     "extra_graph_1":extra_graph_1(),
#     "extra_graph_2":extra_graph_2()
# }
#
# graph_name = list(test_graphs.keys())[0]
#
# g = test_graphs[graph_name]
# pr = scaled_page_rank(graph=g, num_iter = 100, eps=3/7.0)
# #pr = nx.pagerank(nx.DiGraph(g.adjacency_matrix), alpha=1/7.0)
#
# nx.draw(
#     nx.DiGraph(g.adjacency_matrix),
#     labels = dict(zip(pr.keys(),["{} : PR={}".format(node, round(pr[node], 2)) for node in pr])),
#     arrows=True,
#     node_size=[i*100000 for i in list(pr.values())],
#     node_color='b',
#     alpha=0.3
# )
# plt.suptitle("Example " + graph_name)
# plt.show()
########################################################################################################################





# Part 5, bonus
#9
#(a)
epsilon = 0.005
N = 2000
count_of_one = 0

for i in range(N):
    if random.random() < (0.5+epsilon):
        count_of_one += 1

Majority_percent = (count_of_one/N)*100
print(Majority_percent)



#(b)
Total_percent_arr = []
majority_vote = []
N_arr = []

for N in range(5000,105000,5000):
    count_of_ones = 0
    for i in range(N):
        if random.random() < (0.5+epsilon):
            count_of_ones += 1
    if (count_of_ones/N)*100 > 50:
        majority_vote.append(1)
    else:
        majority_vote.append(0)
    Total_percent_arr.append((count_of_ones/N)*100)
    N_arr.append(N)

epsilon_chk = 0.005
N_chk_max = 100000
x = []
y = []

for i in range(N_chk_max):
    x.append(i)
    yy = 1-2*np.exp(-2*epsilon_chk*epsilon_chk*i)
    y.append(yy)

plt.plot(x,y)

for N in range(1,100001, 50):
    count_of_ones = 0
    for i in range(N):
        if random.random() < (0.5+epsilon):
            count_of_ones += 1
    if (count_of_ones/N)*100 > 50:
        majority_vote.append(1)
    else:
        majority_vote.append(0)
    #Total_percent_arr.append((count_of_ones/N)*100)
    
    N_arr.append(N)

win = 0
win_per = []

for x in range(0, 1950):
    win = 0
    for i in range(x, x+50):
        win += majority_vote[i]
    win_per.append(win/50)

plt.plot(range(0,100000,51)[0:1950], win_per)
plt.plot(x,y)








