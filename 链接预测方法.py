
"""
Link prediction algorithms.
"""
# CN
def common_neighbors_index(G, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居
    sim_dict = {}    # 存储相似度的字典

    node_num = nx.number_of_nodes(G)
    for u in range(node_num - 1):
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += 1
                        else:
                            sim_dict[(u,v)] = 1
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    return sim_dict
# end def

# PA
def preferential_attachment_index(G, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居
    ebunch = nx.edges(G) if for_existing_edge else nx.non_edges(G)
    
    sim_dict = {}   # 存储相似度的字典

    degree_list = [nx.degree(G, v) for v in range(G.number_of_nodes())]

    for u, v in ebunch:
        s = degree_list[u] * degree_list[v]
        if s > 0:
            sim_dict[(u, v)] = s
        # end if
    # end for

    return sim_dict
# end def

# Jaccard
def jaccard_coefficient(G, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居
        
    sim_dict = {}   # 存储相似度的字典
    # 首先计算分子上的内容（公共邻居）
    node_num = nx.number_of_nodes(G)
    for u in range(node_num - 1):
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += 1.0
                        else:
                            sim_dict[(u,v)] = 1.0
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    # 然后再计算最后的值
    degree_list = [nx.degree(G, v) for v in range(node_num)]

    for (u, v) in sim_dict.keys():
        s = sim_dict[(u, v)]
        sim_dict[(u, v)] = s / (degree_list[u] + degree_list[v] - s)
    # end for
    
    return sim_dict
# end def

# Salton Cosine index
def cosine(G, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居
        
    sim_dict = {}   # 存储相似度的字典
    # 首先计算分子上的内容（公共邻居）
    node_num = nx.number_of_nodes(G)
    for u in range(node_num - 1):
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += 1.0
                        else:
                            sim_dict[(u,v)] = 1.0
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    # 然后再计算最后的值
    degree_list = [nx.degree(G, v) for v in range(node_num)]

    for (u, v) in sim_dict.keys():
        s = sim_dict[(u, v)]
        sim_dict[(u, v)] = s / math.sqrt(degree_list[u] * degree_list[v])
    # end for
    
    return sim_dict
# end def

# AA
def adamic_adar_index(G, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居

    sim_dict = {}   # 存储相似度的字典   
    node_num = nx.number_of_nodes(G)
    log_degree_list = [(0 if nx.degree(G, v) == 0 else math.log2(nx.degree(G, v))) for v in range(node_num)]

    for u in range(node_num - 1):
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += 1 / log_degree_list[w]
                        else:
                            sim_dict[(u,v)] = 1 / log_degree_list[w]
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    return sim_dict
# end def


# RA
def resource_allocation_index(G, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居

    sim_dict = {}   # 存储相似度的字典    
    node_num = nx.number_of_nodes(G)
    degree_list = [nx.degree(G, v) for v in range(node_num)]

    for u in range(node_num - 1):
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u, v) in sim_dict:
                            sim_dict[(u, v)] += 1 / degree_list[w]
                        else:
                            sim_dict[(u, v)] = 1 / degree_list[w]
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    return sim_dict
# end def


# ADP
def adaptive_degree_penalization(G, alpha, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居

    """
     @Article{martinez2016adaptive,
      author   = {Mart\'inez, V\'ictor and Berzal, Fernando and Cubero, Juan-Carlos},
      title    = {Adaptive degree penalization for link prediction},
      journal  = {Journal of Computational Science},
      year     = {2016},
      volume   = {13},
      pages    = {1 - 9},
      issn     = {1877-7503},
      doi      = {http://dx.doi.org/10.1016/j.jocs.2015.12.003},
      keywords = {Link prediction, Networks, Graphs, Topology, Shared neighbors },
      url      = {http://www.sciencedirect.com/science/article/pii/S187775031530051X},
    }
     $s^{ADP}(u, v) = \sum_{w \in \Gamma_u \cap \Gamma_v} \vert \Gamma_w \vert^{-\beta C}$
    """

    C = nx.average_clustering(G)
    param = -1 * alpha * C

    sim_dict = {}  # 存储相似度的字典
    node_num = nx.number_of_nodes(G)
    value_list = [0 if nx.degree(G, v) == 0 else pow(nx.degree(G, v), param) for v in range(node_num)]

    for u in range(node_num - 1):
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += value_list[w]
                        else:
                            sim_dict[(u,v)] = value_list[w]
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    return sim_dict
# end def


# CN_PA
def CN_PA(G, alpha, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False, 计算不存在的边之间的公共邻居):

    """
    @article{zeng2016link,
    title = "Link prediction based on local information considering preferential attachment ",
    journal = "Physica A: Statistical Mechanics and its Applications ",
    volume = "443",
    number = "",
    pages = "537 - 542",
    year = "2016",
    note = "",
    issn = "0378-4371",
    doi = "http://dx.doi.org/10.1016/j.physa.2015.10.016",
    url = "http://www.sciencedirect.com/science/article/pii/S0378437115008626",
    author = "Shan Zeng",
    keywords = "Link prediction",
    keywords = "Complex networks",
    keywords = "Similarity index",
    keywords = "Node similarity "
    }
    """
            
    sim_dict = {}   # 存储相似度的字典
    # 首先计算公共邻居
    node_num = nx.number_of_nodes(G)
    for u in range(node_num - 1):
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += 1.0
                        else:
                            sim_dict[(u,v)] = 1.0
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    # 然后再计算最后的值
    # 计算每个顶点的度，顶点的平均度和系数
    degree_list = [nx.degree(G, v) for v in range(node_num)]
    avg_degree = sum(degree_list) / node_num
    param = alpha / avg_degree

    ebunch = nx.edges_iter(G) if for_existing_edge else nx.non_edges(G)
    for u, v in ebunch:
        s = param * degree_list[u] * degree_list[v]
        if s > 0:
            if (u, v) in sim_dict:
                sim_dict[(u, v)] += s
            else:
                sim_dict[(u, v)] = s
            # end if
       # end if
    # end for    

    return sim_dict
# end def

# CNaD
def common_neighbors_and_distance(G, max_length):
    """
    @article{yang2016predicting,
      title={Predicting missing links in complex networks based on common neighbors and distance},
      author={Yang, Jinxuan and Zhang, Xiao-Dong},
      journal={Scientific Reports},
      volume={6},
      pages={38208},
      year={2016},
    }
    """
    
    sim_dict = {}  # 存储相似度的字典
    # 首先计算公共邻居
    node_num = nx.number_of_nodes(G)
    for u in range(node_num - 1):
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if not G.has_edge(u, v):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += 1.0
                        else:
                            sim_dict[(u,v)] = 1.0
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    # 然后再计算最后的值
    dist_dict = nx.all_pairs_shortest_path_length(G, max_length)     # Dictionary of shortest path lengths keyed by source and target

    for u, v in nx.non_edges(G):
        if (u, v) in sim_dict:  # 有公共邻居
            s = sim_dict[(u, v)]
            sim_dict[(u, v)] = (s + 1) / 2
        else:
            d = 0
            try:
                d = dist_dict[u][v]
            except KeyError:
                pass
            # end try
            if d > 0:
                sim_dict[(u, v)] = 1 / d
            # end if
        # end if
    # end for

    return sim_dict
# end def

# CAR
def CAR(G, method=1, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居

    """
    @article{cannistraci2013from,
      title={From link-prediction in brain connectomes and protein interactomes to the local-community-paradigm in complex networks},
      author={Cannistraci, Carlo Vittorio and Alanis-Lobato, Gregorio and Ravasi, Timothy},
      journal={Scientific Reports},
      volume={3},
      pages={1613},
      year={2013}
    }

    $CAR(x, y) = CN(x, y) \cdot \sum_{z \in \Gamma(x) \cap \Gamma(y)} \frac{|\gamma(z)|}{2}$
    """
    
    sim_dict = {}  # 存储相似度的字典
    gama_dict = {} # 存储局部社团的个数

    alpha = 0 if method == 1 else 1     # alpha = 0 原始的CAR; alpha = 1 改进的CAR

    node_num = nx.number_of_nodes(G)
    for u in range(node_num - 1):
        u_neighbor_set = set(nx.neighbors(G, u))    # u 的邻居集合
        for w in nx.neighbors(G, u):
            # w_neighbor_set = set(nx.neighbors(G, w))    # w 的邻居集合
            uw_set = u_neighbor_set & set(nx.neighbors(G, w))  # u, w 的公共邻居的集合
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u, v) in sim_dict:
                            gama_dict[(u, v)] += len(uw_set & set(nx.neighbors(G, v)))
                            sim_dict[(u, v)] += 1.0  # 暂时存放公共邻居
                        else:
                            sim_dict[(u, v)] = 1.0
                            gama_dict[(u, v)] = len(uw_set & set(nx.neighbors(G, v)))
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    for key in sim_dict.keys():
        t = (alpha + gama_dict[key]) / 2
        sim_dict[key] *= t       
    # end for

    return sim_dict
# end def


# CRA
def CRA(G, method=1, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居

    sim_dict = {}  # 存储相似度的字典

    alpha = 0 if method == 1 else 1  # alpha = 0 原始的CAR; alpha = 1 改进的CAR
    degree_list = [nx.degree(G, v) for v in range(G.number_of_nodes())]

    node_num = nx.number_of_nodes(G)
    for u in range(node_num - 1):
        u_neighbor_set = set(nx.neighbors(G, u))  # u 的邻居集合
        for w in nx.neighbors(G, u):
            # w_neighbor_set = set(nx.neighbors(G, w))    # w 的邻居集合
            uw_set = u_neighbor_set & set(nx.neighbors(G, w))  # u, w 的公共邻居的集合
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        t = len(uw_set & set(nx.neighbors(G, v)))
                        if t == 0 and alpha == 0: continue
                        if (u, v) in sim_dict:
                            sim_dict[(u, v)] += (alpha + t) / degree_list[w]
                        else:
                            sim_dict[(u, v)] = (alpha + t) / degree_list[w]
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    return sim_dict
# end def

# CAA
def CAA(G, method=1, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居

    sim_dict = {}  # 存储相似度的字典

    alpha = 0 if method == 1 else 1  # alpha = 0 原始的CAR; alpha = 1 改进的CAR
    degree_list = [nx.degree(G, v) for v in range(G.number_of_nodes())]

    node_num = nx.number_of_nodes(G)
    for u in range(node_num - 1):
        u_neighbor_set = set(nx.neighbors(G, u))  # u 的邻居集合
        for w in nx.neighbors(G, u):
            # w_neighbor_set = set(nx.neighbors(G, w))    # w 的邻居集合
            uw_set = u_neighbor_set & set(nx.neighbors(G, w))  # u, w 的公共邻居的集合
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        t = len(uw_set & set(nx.neighbors(G, v)))
                        if t == 0 and alpha == 0: continue
                        if (u, v) in sim_dict:
                            sim_dict[(u, v)] += (alpha + t) / math.log2(degree_list[w])
                        else:
                            sim_dict[(u, v)] = (alpha + t) / math.log2(degree_list[w])
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for

    return sim_dict
# end def


# CCLP
def CCLP(G, method=1, for_existing_edge=False):
    # for_existing_edge=True 计算存在的边之间的公共邻居
    # for_existing_edge=False 计算不存在的边之间的公共邻居

    """
    @article{wu2016link,
    title = "Link prediction with node clustering coefficient",
    journal = "Physica A: Statistical Mechanics and its Applications",
    volume = "452",
    number = "Supplement C",
    pages = "1 - 8",
    year = "2016",
    issn = "0378-4371",
    doi = "https://doi.org/10.1016/j.physa.2016.01.038",
    url = "http://www.sciencedirect.com/science/article/pii/S0378437116000777",
    author = "Zhihao Wu and Youfang Lin and Jing Wang and Steve Gregory",
    keywords = "Link prediction",
    keywords = "Complex networks",
    keywords = "Clustering coefficient"
    }

    $CCLP(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} CC_z$
    """

    node_num = nx.number_of_nodes(G)
    cc_list = [nx.clustering(G, v) for v in range(node_num)]  # 存放每个节点的聚集系数

    alpha = 0 if method == 1 else 1  # alpha = 0 原始的CAR; alpha = 1 改进的CAR

    sim_dict = {}  # 存储相似度的字典
   
    for u in range(node_num - 1):
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += alpha + cc_list[w]
                        else:
                            sim_dict[(u,v)] = alpha + cc_list[w]
                        # end if
                    # end if
                # end if
            # end for
        # end for
    # end for
    
    return sim_dict
# end def

def LNB(G, method):
    """
    @article{Liu2011Link,
      author={Zhen Liu and Qian-Ming Zhang and Linyuan L\"{u} and Tao Zhou},
      title={Link prediction in complex networks: A local naïve Bayes model},
      journal={EPL (Europhysics Letters)},
      volume={96},
      number={4},
      pages={48007},
      url={http://stacks.iop.org/0295-5075/96/i=4/a=48007},
      year={2011}
    }
    """

    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)
    M = node_num * (node_num - 1) / 2
    s = M / edge_num - 1
    logs = math.log2(s)

    degree_list = [nx.degree(G, v) for v in range(node_num)]
    # 计算每个顶点的role
    log_role_list = [nx.triangles(G, w) for w in range(node_num)]   # 三角形个数
    for w in range(node_num):
        triangle = log_role_list[w]
        numerator = triangle + 1
        d = degree_list[w]
        non_triangle = d * (d - 1) / 2 - triangle
        denominator = non_triangle + 1

        log_role_list[w] = math.log2(numerator / denominator) + logs
    # end for

    sim_dict = {}  # 存储相似度的字典

    # 计算相似度
    for u in range(node_num - 1):
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                # w 是 u，v的公共邻居
                if u < v and not G.has_edge(u, v):
                    if method == 'CN':
                        s = log_role_list[w]
                    elif method == 'AA':
                        s = 1 / math.log2(degree_list[w]) * log_role_list[w]
                    else:   # RA
                        s = 1 / degree_list[w] * log_role_list[w]
                    # end if
                    if (u, v) in sim_dict:
                        sim_dict[(u, v)] += s
                    else:
                        sim_dict[(u, v)] = s + node_num  # 防止出现负值
                    # end if
                # end if
            # end for
        # end for
    # end for   

    return sim_dict
# end def


# LP
def local_path_index(G, alpha):

    sim_dict = {}  # 存储相似度的字典

    node_num = nx.number_of_nodes(G)
    # a--b--c--d
    for a in range(node_num - 1):
        for b in nx.neighbors(G, a):
            for c in nx.neighbors(G, b):
                # b 是 a，c的公共邻居
                if a < c and not G.has_edge(a, c):
                    if (a, c) in sim_dict:
                        sim_dict[(a, c)] += 1
                    else:
                        sim_dict[(a, c)] = 1
                    # end if
                # end if
                # 继续查找长度为3的路径
                for d in nx.neighbors(G, c):
                    if a < d and not G.has_edge(a, d):
                        if (a, d) in sim_dict:
                            sim_dict[(a, d)] += alpha
                        else:
                            sim_dict[(a, d)] = alpha
                        # end if
                    # end if
                # end for
            # end for
        # end for
    # end for

    return sim_dict
# end def

