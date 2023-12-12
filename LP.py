import pulp, json, itertools, time
import networkx as nx
import pandas as pd
import numpy as np

class LP():
    def __init__(self, G: nx.DiGraph, weight_name: str, type_name: str) -> None:
        self.G = G.copy()
        self.weight_name = weight_name
        self.type_name = type_name

        self.H = G.copy().to_undirected()
        self.nodes, self.G_edges, self.H_edges = list(self.G.nodes), list(self.G.edges()), list(self.H.edges())
        self.G_weight = {(u,v): attr[self.weight_name] for u,v,attr in self.G.edges(data=True)}

    def solve_LP(self) -> pd.DataFrame:
        # declare variables: x(u,v), y(u,v), r(u)
        agony_x = { 
            edge: pulp.LpVariable("x_{}_{}".format(*edge), cat='Integer') 
                for edge in self.G_edges
        }
        agony_y = { 
            edge: pulp.LpVariable("y_{}_{}".format(*edge), cat='Integer') 
                for edge in self.H_edges        
        }
        ranking = {
            node: pulp.LpVariable("r{}".format(node), cat='Integer') 
                for node in self.nodes
        }
        
        # declare the LP problem
        LP = pulp.LpProblem("Crime_Ranking", sense=pulp.LpMinimize)

        # declare the objective function
        LP += pulp.lpSum(list(agony_x.values()) + list(agony_y.values()))

        # add constraints (1) and (2)
        for (u,v) in self.G_edges:
            LP += (agony_x[(u,v)] >= (ranking[u] - ranking[v] + 1)*self.G_weight[(u,v)])
            LP += (agony_x[(u,v)] >= 0)

        # add constraints (3) and (4)
        for (u,v) in self.H_edges:
            LP += (agony_y[(u,v)] >= (ranking[u] - ranking[v]))
            LP += (agony_y[(u,v)] >= (ranking[v] - ranking[u]))

        # add constraints (5)
        for n in self.nodes:
            LP += (ranking[n]>=1)

        # 開始求解
        LP.solve()
        pulp.LpStatus # 所有狀態
        LP.status # 此次求解狀態
        
        dct = {v.name: [v.varValue] for v in LP.variables() if "r" in v.name}
        df = pd.DataFrame.from_dict(dct, orient='index')
        df.columns = ["ranking_{}".format(self.type_name)]
        
        return df.sort_values(by=["ranking_{}".format(self.type_name)])


