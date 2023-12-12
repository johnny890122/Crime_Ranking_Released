import pulp, json, itertools, time
import networkx as nx
import pandas as pd
import numpy as np
from LP import LP

# 將通話記錄轉換成 LP 所需的形式
def to_directed(df: pd.DataFrame) -> pd.DataFrame: 
    cross_df = pd.crosstab(df["source"], df["target"], values=df["weight"], aggfunc='sum')
    
    idx = cross_df.columns.union(cross_df.index)
    adj_mat = cross_df.reindex(index = idx, columns=idx).fillna(0)

    tmp = adj_mat - adj_mat.T
    tmp[tmp<0] = 0

    directed_df = tmp.stack()
    directed_df = directed_df.to_frame().reset_index()
    directed_df.columns = ["source", "target", "weight"]
    directed_df = directed_df[directed_df["weight"] != 0 ]

    directed_df["log2(weight)"] = np.log2(directed_df["weight"])
    
    return directed_df

def rank_by_order(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # 保留下「不是自己v.s.自己」&「通話紀錄」
    df = df[ (df["Role_From"] != df["Role_Received"]) & (df["Type"] == "Voice")]

    # 根據「指定的頻率」進行分組。
    if freq == "date":
        group_df = df.groupby([df['Time'].dt.date]).ngroup().to_frame().rename(columns= {0: "group_id"})
    elif freq == "half_day":
        group_df = df.groupby([df['Time'].dt.date, df['Time'].dt.strftime('%p')]).ngroup().to_frame().rename(columns= {0: "group_id"})
    elif freq == "hour":
        group_df = df.groupby([df['Time'].dt.hour]).ngroup().to_frame().rename(columns= {0: "group_id"})
    else:
        raise Exception("Sorry, {} is not implemented.".format(freq))
    df = df.join(group_df)
    # 計算該組的組間數量
    df = df.merge(df.groupby("group_id").size().to_frame(), on="group_id")
    df = df.rename(columns = {0: "total_call_by_period"})

    # 進行排名
    df["order_by_period"] = df.sort_values(
                                ["group_id", "Time"], ascending=False
                            ).groupby(["group_id"])["Time"].cumcount() + 1

    df["score"] = df["order_by_period"] / df["total_call_by_period"]

    cluster_df = df.groupby(
        [df['Role_From'], df['Role_Received']])["score"].sum().to_frame().reset_index()
    cluster_df.columns = ["source", "target", "weight"]
    
    cluster_df["log2(weight)"] = np.log2(cluster_df["weight"])
    
    return cluster_df

def randomzie(df: pd.DataFrame) -> pd.DataFrame:
    sub_df = df[["ID_From", "ID_Received"]].reset_index(drop=False).rename(columns={"index": ""})

    # 先將 row random shuffle
    sub_df = sub_df.sample(frac=1)
    sub_df["group_id"] = np.array(sub_df.index // 2, dtype=int)

    # 決定哪些 edges 要 rewire & flip
    whether_rewire_df = pd.DataFrame(np.random.rand(len(sub_df.index)//2 + 1) > 0.5).reset_index(drop=False)
    whether_rewire_df.columns = ["group_id", "whether_rewire"]
    whether_flip_df = pd.DataFrame(np.random.rand(len(sub_df.index)) > 0.5).reset_index(drop=False)
    whether_flip_df.columns = ["group_id", "whether_flip"]
    sub_df = sub_df.merge(whether_rewire_df, on="group_id").merge(whether_flip_df, on="group_id")

    # 將要 rewire 的 edges swap 
    sub_df['ID_Received_tmp'] = sub_df[sub_df["whether_rewire"]].groupby('group_id')["ID_Received"].transform(np.roll, shift=1)
    sub_df.loc[~sub_df["whether_rewire"], 'ID_Received_tmp'] = sub_df[~sub_df["whether_rewire"]]["ID_Received"]
    sub_df["ID_Received"] = sub_df['ID_Received_tmp']

    sub_df[['ID_From', 'ID_Received']] = sub_df[['ID_Received', 'ID_From']].where(sub_df['whether_flip'], sub_df[['ID_From','ID_Received']].values)
    sub_df = sub_df.set_index("")
    sub_df = sub_df[["ID_From", "ID_Received"]]
    return df.drop(["ID_From", "ID_Received"], axis=1).join(sub_df)

def phone_mapping(df: pd.DataFrame) -> pd.DataFrame:
    role = pd.read_excel("input/腳色.xlsx", sheet_name="重要號碼輸出")
    phone_dct = {phone: role for (phone, role) in zip(role["號碼"], role["角色"])}
    df["Role_From"] = df["ID_From"].apply(lambda x: phone_dct[x])
    df["Role_Received"] = df["ID_Received"].apply(lambda x: phone_dct[x])
    
    return df

def solve_LP_static_G(df: pd.DataFrame) -> pd.DataFrame:
    cluster_df = df.groupby([df['Role_From'], df['Role_Received']])["nr"].count().to_frame().reset_index()
    cluster_df.columns = ["source", "target", "weight"]
    static_df = to_directed(cluster_df)
    G = nx.from_pandas_edgelist(static_df, edge_attr=True, create_using=nx.DiGraph())
    Q = LP(G, "weight", "static")
    return Q.solve_LP()

def solve_LP_temporal_G(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    temporal_df = to_directed(rank_by_order(df, freq))
    G = nx.from_pandas_edgelist(temporal_df, edge_attr=True, create_using=nx.DiGraph())

    Q = LP(G, "weight", "temporal_{}".format(freq))
    return Q.solve_LP()
