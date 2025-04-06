import numpy as np
import torch
import random
import scipy.sparse
from utils.utils import resample_branch_by_step

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_weight(model, weight, reg_model=None):
    model.load_state_dict(weight['VAE'])
    if 'regression' in weight and reg_model is not None:
        reg_model.load_state_dict(weight['regression'])

def edge_calculation(dataset, size=256):
    rows = []
    cols = []
    data = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i][1])): #for ith branch and its prefix branches (self incl)
            rows.append(dataset[i][1][j])
            cols.append(dataset[i][0][-1])
            data.append(len(dataset[i][0]))
    edge = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
    return edge

def node_calculation(layer, node):
    """
    pre_node:
    #size (num_soma_branches, #descendants_from_soma_branch)
    #[list of branch ids rooted at root branch 0],[.. at root branch 1], ...] 

    Function: 
    seems to be getting all the ancestor branches (up until the same depth as current branch) 
    in the branch substree that this branch is in 
    """
    #layer = [(root_id, depth)]
    #node: forest map of substrees rooted at each soma branch
    pre_node = [[] for i in range(len(layer))] #(num_soma_branches, #descendants_from_soma_branch):[[list of branch ids rooted at root branch 0],[.. at root branch 1], ...] 
    for i in range(len(layer)): #consider each branch and its soma branch
        for depth in range(layer[i][1]): #up to depth before depth of cur branch
            pre_node[i] += node[layer[i][0]][depth] #PROBLEM: include non-ancetor branches rooted at same soma branch -> global topology? 
    return pre_node

def tree_construction(branches, dataset, layer, nodes):
    """
    node: [no_prev_layers_branches x L x 3]
    for each branch in tree: node = [[soma branch coords at layer 0], [branch 1 at layer 1], [b2 at layer 1], ...]
    """
    #construct trees for global topology 
    # var node = a subtree (set of branches). Size: (num_branches, branch_len, 3)
    #we build a tree rooted at each branch, but for non-soma branches, their tree structure will be empty
    branches = np.array(branches)
    e = edge_calculation(dataset, size=len(branches))
    pre_node = node_calculation(layer, nodes) # branch i gets an entry in list layer (as (root_id, depth)) at the ith idx
    tree = []
    for i in range(len(branches)):
        node = branches[pre_node[i]] # find all branches that are in previous layers & share same soma branch as branch i
        #build an adjacency matrix: size: num_branches x num_branches (tree graph rep)
        m, n = np.ix_(pre_node[i], pre_node[i])
        #convert into coordinate sparse format: efficient use in PyTorch sparse tensors and for passing into a T-GNN
        edge = e[m, n].tocoo() 
        # a tree of subtrees (a forest of trees for each soma branch): 
        # node is a subtree i.e. list of nodes (branches), edge = adjacency matrix of that subtree 
        tree.append({'edge': edge, 'node': node})
    print("[CHECK] tree[-1]['node']", tree[-1]['node']) #expect [no.prev.branches x L x 3]
    return tree

def my_collate(data):
    """
    data is a batch of samples: Each data[i] contains a single sample:
    (
    padded_source,    # prefix branches [W, L, 3]
    target_l,         # left child branch [L, 3]
    target_r,         # right child branch [L, 3]
    real_wind_len,    # int
    seq_len,          # [W] (stores length of each prefix branch, W prefix branches)
    target_len,       # [2] (length of left and right branches)
    node,             # [#branches, L, 3] (node-level global condition)
    edge              # sparse edge matrix (adjacency for global condition)
    )
    
    return output: 
    (
    padded_source,  # [B, W, L, 3]
    target_l,       # [B, L, 3]
    target_r,       # [B, L, 3]
    real_wind_len,  # [B]
    seq_len,        # [B, W]
    target_len,     # [B, 2]
    node,           # [total_nodes, L, 3]
    offset,         # [total_nodes]
    edge            # sparse_coo_tensor
    )
    * B=batch size 
    * offset: specifies which data sample each branch in node ds belongs to 
    * total_nodes = total num branches in batch
    #nodes -> list of branches [L,3], each branch is viewed as a node in tree graph rep (flattend across batch)
    #edge 



    Conditional info: 
    Local condition: padded_source (W prefix branches of shape [L, 3])
	Global condition: node, edge, and offset
    """
    #after sampling a batch data, combine them together in the follow way
    #see ConditionalPrefixSeqDataset get_item for what each var below means
    padded_source = torch.stack([data[i][0] for i in range(len(data))], dim=0)
    target_l = torch.stack([data[i][1] for i in range(len(data))], dim=0)
    target_r = torch.stack([data[i][2] for i in range(len(data))], dim=0)
    real_wind_len = torch.stack([torch.tensor(data[i][3]) for i in range(len(data))], dim=0)
    seq_len = torch.stack([data[i][4] for i in range(len(data))], dim=0)
    target_len = torch.stack([data[i][5] for i in range(len(data))], dim=0)

    node = [data[i][6] for i in range(len(data))] #each data[i] is about same branch (batchsize, #branches per sample, L, 3)
    offset = []
    for i in range(len(data)):
        offset += [i for j in range(len(data[i][6]))] # 1D array: for each branch in node (dim 2), record which data sample it belongs to 

    if offset == []:
        offset = torch.tensor([])
        print("[DEBUG] data_utils.py: if offset==[]")
    elif offset[-1] != len(data) - 1:
        offset.append(len(data) - 1)
        # node.append(torch.zeros((1, 16, 3))) 
        node.append(torch.zeros((1,32,3)))
        print("[DEBUG] data_utils.py: else offset[-1]")
    # print("[DEBUG] data_utils.py: node", node)
    offset = torch.tensor(offset)
    node = torch.concat(node, dim=0) # (batch_size * # branches per sample, L,3)
    
    # [ edge_0   0        0      ] #edge_0 and 0 are matrices 
    # [   0    edge_1     0      ]
    # [   0      0      edge_2   ]
    # no interaction across different samples (each sample is a branch, and the subtrees rooted at the branch if it is a soma branch) 
    edge = scipy.sparse.block_diag(mats=[data[j][7] for j in range(len(data))]) #(total_nodes x total_nodes)
    layer = edge.data #value of non-zero elems in edge sparse matrix
    row = edge.row #row index of corresponding value in edge sparse matrix 
    col = edge.col #col index ... 
    #so at index i we have: edge[row[i], col[i]] has value == layer[i]
    #just marking connection here - actual edge connection vals understand edge_calculation
    data = np.ones(layer.shape) #data var got renamed 
    if edge.shape[0] == 0 or edge.shape[1] == 0:
        _max = 0
    else:
        _max = edge.max()
    shape = (int(_max + 1), edge.shape[0], edge.shape[1])
    edge = torch.sparse_coo_tensor(torch.tensor(np.vstack([layer, row, col])).to(torch.long), torch.tensor(data), shape)
    return (padded_source, target_l, target_r, real_wind_len, seq_len, target_len, node, offset, edge)

class ConditionalPrefixSeqDataset(torch.utils.data.Dataset):
    def __init__(
            self, branches, dataset,
            max_src_length, max_dst_length, data_dim, max_window_length, trees, max_size=128,
            masking_element=0, resample=True
    ):
        self.branches = branches
        self.dataset = dataset
        self.max_src_length = max_src_length #input branch len
        self.max_dst_length = max_dst_length #predicted output branch len
        self.max_window_length = max_window_length #w parameter for branch smoothing? 
        self.data_dim = data_dim # dim of 3D coordiante 
        self.masking_element = masking_element
        self.resample = resample
        self.trees = trees
        self.max_size = max_size
        self.resampled_branches = [resample_branch_by_step(branch, self.max_dst_length, len(branch)) for branch in
                                branches] if self.resample else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        neuron agnostic -> example for training is a bifurcation where
        with all prefix branches (limited by window len) and the 2 children branches. 
        The prefix branches for the h_local input seq.

        This function deals with a single branch at index=index. 

        Returning: information about a single branch
        padded_source: prefix branches up to bifurcation (i.e. the local prefix path)
        target_l: target left child branch
        target_r: target right child branch
        real_wind_len: prefix window i.e. num of prefix branches to consider 
        seq_len: list of branch len for each selected prefix branch
        target_len: len of left child and right child branches
        node: list of branches rooted at "soma" branch each of size [L, 3]
            with branch_id = index (empty if branch at index is not a soma branch)
        edge: adjacency matrix for all the branches in subtree node. 
        """
        #get prefix branches (i.e. req for local condition) for a single branch (idx) from a single neuron
        wind_l = self.max_window_length
        #source shape: [wind_l, max_src_L, 3]
        #wind_l = number prefix branches considered in learning local condition 
        #pading for branches shorter than max src length
        s_shape = (wind_l, self.max_src_length, self.data_dim)
        padded_source = torch.ones(s_shape) * self.masking_element
        #target shape: [L, 3] where L = # nodes in a branch
        t_shape = (self.max_dst_length, self.data_dim)
        target_l = torch.ones(t_shape) * self.masking_element
        target_r = torch.ones(t_shape) * self.masking_element
        #real_wind_len = actual # prefix branches to current branch (incl cur)
        real_wind_len, seq_len = 0, []
        # prefix = prefix branches leading up to the children branches, targets = 2 children branches 
        prefix, targets, _ = self.dataset[index]
        #select the last wind_l number of prefix branches from the prefix list 
        #CREATE padded prefix branch list (incl. direct parent branch/cur branch to children)
        for idx, branch_id in enumerate(prefix[-wind_l:]): #for each prefix branch (we select at most wind_l num) to current branch (incl)
            branch = torch.from_numpy(self.branches[branch_id])
            branch_l = len(branch)
            padded_source[idx][:branch_l] = branch
            real_wind_len += 1 #case for # prefix branch < wind_l (terminate loop early)
            seq_len.append(branch_l) #seq_len=list of branch len for selected prefix branches

        while len(seq_len) != wind_l:
            seq_len.append(0)
        seq_len = torch.LongTensor(seq_len)
        if self.resample:
            #resample target branches to max_dst_length
            target_l = torch.from_numpy(self.resampled_branches[targets[0]]).to(torch.float32)
            target_r = torch.from_numpy(self.resampled_branches[targets[1]]).to(torch.float32)
            target_len = torch.tensor([self.max_dst_length, self.max_dst_length])
        else:
            #keep original branch length, pad shorter branch with 0 until all reach max_dst_length
            branch_l, branch_r = self.branches[targets[0]], self.branches[targets[1]]
            target_len = [len(branch_l), len(branch_r)]
            target_l[:target_len[0]] = torch.from_numpy(branch_l)
            target_r[:target_len[1]] = torch.from_numpy(branch_r)

        new_index = prefix[-1] #last prefix branch
        #it's possible that if index doesn't denote a soma branch then node and edge ds will be empty 
        # print("[DEBUG] self.trees[new_index]:", self.trees[new_index])
        # print("[DEBUG] self.trees[new_index][node]:", self.trees[new_index]["node"], "new_index:", new_index)
        node = torch.from_numpy(self.trees[new_index]['node'])
        node = node.to(torch.float32)
        # print("[DEBUG] data_utils.py: new_index, node.shape", new_index, node.shape)
        # if node.shape[0] == 0:
        #     print("[DEBUG] data_utils new_index, node (soma branch):", new_index, node)
        edge = self.trees[new_index]['edge']
        
        #target_len = [len(left child branch), len(right child)]
        return padded_source, target_l, target_r, real_wind_len, seq_len, target_len, node, edge



class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, reverse=True):
        super(Seq2SeqDataset, self).__init__()
        self.data, self.seq_len = data, seq_len
        self.reverse = reverse

    def __len__(self):
        return len(self.seq_len)

    def __getitem__(self, index):
        t_len = self.seq_len[index]
        target = self.data[index][:t_len]
        source = target[::-1] if self.reverse else target
        assert len(source) == len(target), \
            'source and target are not of the same length'
        return source, target, t_len


def get_seq_to_seq_fn(masking_element, output_dim):
    def col_fn(batch):
        max_len, batch_size, tls = max(x[2] for x in batch), len(batch), []
        pad_src = np.ones((batch_size, max_len, output_dim)) * masking_element
        pad_tgt = np.ones((batch_size, max_len, output_dim)) * masking_element

        for idx, (src, tgt, tl) in enumerate(batch):
            tls.append(tl)
            pad_src[idx, :tl] = src
            pad_tgt[idx, :tl] = tgt

        pad_src = torch.from_numpy(pad_src).float()
        pad_tgt = torch.from_numpy(pad_tgt).float()

        return pad_src, pad_tgt, tls
    return col_fn



def fetch_walk_fix_dataset(neurons, seq_len, reverse, verbose=False):
    all_walks = []
    for neu in (tqdm(neurons, ascii=True) if verbose else neurons):
        curr_walks = neu.fetch_all_walks()
        all_walks.extend([
            resample_branch_by_step(x, seq_len, len(x))
            for x in curr_walks
        ])
    seq_lens = [seq_len] * len(all_walks)
    return Seq2SeqDataset(data=all_walks, seq_len=seq_lens, reverse=reverse)