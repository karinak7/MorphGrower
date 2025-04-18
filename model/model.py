import torch
import random
from torch.nn.utils.rnn import pack_padded_sequence
from utils.debug_config import DEBUG

def reconstruction_loss(reconstructed_x, x, ignore_element=0):
    # reconstruction loss
    # x = [trg len, batch size * n walks, output dim] when tree major
    # x = [trg len, batch size, output dim] when batch major

    seq_len, batch_size, output_dim = x.shape
    mask = x[:, :, 0] != ignore_element
    rec_loss = 0
    # print(torch.all(mask != torch.isinf(x[:, :, 0])))
    for d in range(output_dim):
        # print(reconstructed_x[:, :, d][mask])
        # print(x[:, :, d][mask])
        rec_loss += torch.nn.functional.mse_loss(
            reconstructed_x[:, :, d][mask],
            x[:, :, d][mask], reduction='sum'
        )
        # print(rec_loss)
    return rec_loss / output_dim

class ConditionalSeqEncoder(torch.nn.Module):
    # branch encoder
    # encode one branch into states of
    # hidden & cell (both [n_layers,hidden_dim])
    # Same as the SeqEncoder
    def __init__(
            self, input_dim, embedding_dim,
            hidden_dim, n_layers=2, dropout=0.5
    ):
        super(ConditionalSeqEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers #2 by default
        self.embedding_dim = embedding_dim
        self.coordinate2emb = torch.nn.Linear(input_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=dropout
        )

    def forward(self, src, seq_len):
        """
        src: 
        [[[b1_p0_x, b1_p0_y, b1_p0_z], [b2_p0_x, b2_p0_y, b2_p0_z],[b3_p0_x, b3_p0_y, b3_p0_z], ...], 
            [point 2 on all prefix branches ], 
            [point 3 on all prefix branches], ...]
         each column is a branch, each row is a timestep.
         By default: pytorch lstm takes column as the batch dim (dim=1), dim=0 as the sequence length dim 

        hidden:
        [[[hidden cell for branch1], [hidden cell for branch 2], ...], [[hidden cell2 for branch1], ...]]
        """
        input_seq = self.dropout_fun(self.coordinate2emb(src))
        #input_seq: [seq_len, batch_size, embedding_dim=hidden_dim] #seq_len = number points on a branch L = 32
        packed_seq = pack_padded_sequence(
            input_seq, seq_len, enforce_sorted=False
        )
        #seq_len: Each value tells the actual number of valid timesteps for each branch
        #packed_seq: give LSTM a batch seqs, and tells it their real length before padding so lstm can handle efficiently
        #seq len corresponds to length of each column in input_seq which represents a branch sample
        packed_output, (hidden, cell) = self.rnn(packed_seq)

        # outputs = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        return hidden, cell


class ConditionalSeqDecoder(torch.nn.Module):
    # Same as the SeqDecoder
    def __init__(
            self, output_dim, embedding_dim, hidden_dim,
            n_layers=2, dropout=0.5
    ):
        super(ConditionalSeqDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hid2coordinate = torch.nn.Linear(hidden_dim, output_dim)
        self.coordinate2emb = torch.nn.Linear(output_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=dropout
        )

    def forward(self, init, hidden, cell):
        init = init.unsqueeze(0)
        # init = [1, batch_size, output_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        embedding = self.dropout_fun(self.coordinate2emb(init))
        # print("embedding",embedding.shape)
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        prediction = self.hid2coordinate(output).squeeze(0)
        return prediction, hidden, cell


# target_len 重采样到 max_dst_len
def conditional_decode_seq(
        decoder, output_shape, init, hidden, cell,
        device, teacher_force=0.5, target=None
):
    if teacher_force > 0 and target is None:
        raise NotImplementedError(
            'require stadard sequence as input'
            'when using teacher force'
        )
    target_len, batch_size, output_dim = output_shape
    outputs = torch.zeros(output_shape).to(device)
    current, outputs[0] = init, init
    # print('init',init.shape, init)
    for t in range(1, target_len):
        output, hidden, cell = decoder(current, hidden, cell)
        # print('output',output.shape)
        outputs[t] = output
        current = target[t] if random.random() < teacher_force else output
        # print('current',current.shape)

    return outputs

class ConditionEncoder(torch.nn.Module):
    #path encoding (local info): use LSTM instead of EMA in paper
    def __init__(self, branch_encoder, hidden_dim, n_layers=2, dropout=0.5):
        super(ConditionEncoder, self).__init__()
        self.branch_encoder = branch_encoder
        self.path_rnn = torch.nn.LSTM(
            branch_encoder.n_layers * branch_encoder.hidden_dim * 2,
            hidden_dim, n_layers, dropout=dropout
        )
        self.hidden_dim, self.n_layers = hidden_dim, n_layers

    def forward(self, prefix, seq_len, window_len):
        # prefix = [bs, window len, seq_len, data_dim] -> batch of branch sequences 
        # seq_len = [bs, window len]
        # window_len = [bs]
        bs, wind_l, seq_l, input_dim = prefix.shape
        all_seq_len, all_seq = [], [] #all_seq_len = len of each seq = # branches per seq
        for idx, t in enumerate(window_len):
            all_seq_len.extend(seq_len[idx][:t])
            all_seq.append(prefix[idx][:t])
        all_seq = torch.cat(all_seq, dim=0).permute(1, 0, 2) # [seq_len, batch_size, input_dim]
        # print('[info] seq_shape', all_seq.shape, sum(window_len))

        h_branch, c_branch = self.branch_encoder(all_seq, all_seq_len)
        # print('[info] hshape', h_branch.shape, c_branch.shape)

        hidden_seq = torch.cat([h_branch, c_branch], dim=0)
        # print('[info] hidden_seq_shape', hidden_seq.shape)
        seq_number = sum(window_len)
        inter_dim = self.branch_encoder.n_layers * \
            self.branch_encoder.hidden_dim * 2
        hidden_seq = hidden_seq.transpose(0, 1).reshape(seq_number, -1)
        all_hidden = torch.zeros((bs, wind_l, inter_dim)).to(hidden_seq)
        curr_pos = 0
        for idx, t in enumerate(window_len):
            all_hidden[idx][:t] = hidden_seq[curr_pos: curr_pos + t]
            curr_pos += t
        assert curr_pos == seq_number, 'hidden vars dispatched error'

        all_hidden = all_hidden.permute(1, 0, 2)
        # print('[info] all hidden shape', all_hidden.shape, len(window_len))
        packed_wind = pack_padded_sequence(
            all_hidden, window_len, enforce_sorted=False
        )
        _, (h_path, c_path) = self.path_rnn(packed_wind)
        return h_path, c_path


class ConditionalSeq2SeqVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, distribution, tgnn, device, forgettable=None, remove_path=False,
                remove_global=False, new_model=False, dropout=0.1):
        super(ConditionalSeq2SeqVAE, self).__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.distribution = distribution #vMF passed in 
        self.new_model = new_model
        if new_model:
            self.condition_encoder = ConditionEncoder(self.encoder, self.encoder.hidden_dim, self.encoder.n_layers, dropout=dropout)
        # forgettable == None mean pooling
        # otherwise h_path[w] = forgettable * h_path[w-1]
        #                       + (1 - forgettable) * raw_embedding[w-1]
        self.forgettable = forgettable if forgettable != 0 else None

        #ablation
        print("**************************************")
        print(remove_global, remove_path)
        print("**************************************")

        self.tgnn = tgnn.to(device)
        self.global_dim = self.tgnn.size
        self.remove_global = remove_global
        self.remove_path = remove_path

        #prior distribution? 
        mean = torch.full([1,encoder.hidden_dim],0.0)
        std = torch.full([1,encoder.hidden_dim],1.0)
        self.gauss = torch.distributions.Normal(mean, std)

        #obtain conditioned feature rep via concat & linear projection
        
        #encoder state -> Latent z 
        # encoder.hidden_dim * encoder.n_layers * 6  = CONCAT[r_b1, r_b2, h_local] 
        #r_bn = encoder.hidden_dim * 2 * encoder.n_layers (*2 = pair of hidden and cell states)
        # h_local = encoder.hidden_dim * 2 * encoder.n_layers (EMA of branch reps each of same dim: encoder.hidden_dim * 2 * encoder.n_layers)
        self.state2latent = torch.nn.Linear(
            encoder.hidden_dim * encoder.n_layers * 6 + self.global_dim,
            distribution.lat_dim
        )

        #latent z -> left branch feature rep 
        # input dim = CONCAT[z, h_local, h_global]
        # output dim = pairs of hidden and cell states across n_layers of decoder LSTM. CONCAT[h, c]*encoder.n_layers
        self.latent2state_l = torch.nn.Linear(
            distribution.lat_dim + encoder.hidden_dim * encoder.n_layers * 2 + self.global_dim,
            decoder.hidden_dim * decoder.n_layers * 2
        )
        #latent z -> right branch feature rep
        self.latent2state_r = torch.nn.Linear(
            distribution.lat_dim + encoder.hidden_dim * encoder.n_layers * 2 + self.global_dim,
            decoder.hidden_dim * decoder.n_layers * 2
        )
        assert encoder.hidden_dim == decoder.hidden_dim, \
            "hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "encoder and decoder must have equal number of layers!"

    def encode(self, src_l, seq_len_l, src_r, seq_len_r, h_path, h_global):
        hidden_l, cell_l = self.encoder(src_l, seq_len_l)
        """
        encode logic: 
        1.for left and right ref branches:
            encode branch with LSTM and  hidden & cell state -> encoded state 
        2.Concat ref branch encoding with h_local=h_path and h_global -> conditional state
        -> linear projection -> h
        3. create vMF distribution with h as mean 
        4. sample latent variables Z via avg. of 5 rejection sampling results 

        Paper section: 
        #μ = W · CONCAT[rb1 , rb2 , hglobal, hlocal]
        # zi ∼ vMF(μ, κ)
        # Z=avg(5 samples of z)
        """
        hidden_l, cell_l = self.encoder(src_l, seq_len_l)
        n_layers, batch_size, hid_dim = hidden_l.shape
        states_l = torch.cat((hidden_l, cell_l), dim=0) #[2*n_layers, bs, h_dim] = [4, 256, 64]

        if DEBUG: 
            print(f"[SHAPE] model: ConditionalSeq2SeqVAE.encode: src_l.shape {src_l.shape}")
            print(f"[SHAPE] model: ConditionalSeq2SeqVAE.encode: seq_len_l.shape {seq_len_l.shape}")
            print(f"[SHAPE] model: ConditionalSeq2SeqVAE.encode: hidden_l.shape {hidden_l.shape}")
            print(f"[SHAPE] model: ConditionalSeq2SeqVAE.encode: cell_l.shape {cell_l.shape}")
            print(f"[SHAPE] model: ConditionalSeq2SeqVAE.encode: states_l.shape {states_l.shape}")
            print(f"[SHAPE] model: ConditionalSeq2SeqVAE.encode: h_path.shape {h_path.shape}")
            print(f"[TENSOR] model: ConditionalSeq2SeqVAE.encode: states_l {states_l}")
        # result states = [bs, 2*n_layers*hidden_dim] - group state by batch
        # [bs, infer rest of size into 1 dim] = [bs, 2*n_layers*hidden_dim] = [256, 256]
        states_l = states_l.permute(1, 0, 2).reshape(batch_size, -1)
        if DEBUG: 
            print(f"[SHAPE] model: ConditionalSeq2SeqVAE.encode: states_l.shape after reshape {states_l.shape}")
            print(f"[TENSOR] model: ConditionalSeq2SeqVAE.encode: states_l after reshape {states_l}")

        hidden_r, cell_r = self.encoder(src_r, seq_len_r)
        states_r = torch.cat((hidden_r, cell_r), dim=0)
        # result states = [bs, 2*n_layers*hidden_dim]
        states_r = states_r.permute(1, 0, 2).reshape(batch_size, -1)

        # 拼接上 h_path
        # 拼接上TGN output h_global，shape = [bs,self.global_dim]
        states = torch.cat((states_l, states_r, h_path, h_global), dim=1)
        h = self.state2latent(states) 
        tup, kld, vecs = self.distribution.build_bow_rep(h, n_sample=5)
        Z = torch.mean(vecs, dim=0)
        condition = h_global
        if DEBUG: 
            print(f"[SHAPE] model: ConditionalSeq2SeqVAE.encode: h.shape {h.shape}")
            print(f"[SHAPE] model: ConditionalSeq2SeqVAE.encode: Z.shape {Z.shape}")
            print(f"[SHAPE] model: ConditionalSeq2SeqVAE.encode: condition.shape {condition.shape}")
        #vMF mean, sampled latent variables, h_global 
        return h, Z, condition

    def _get_decoder_states(self, latent, batch_size, h_path, h_global, decode_left):
        # cat latent with h_path and h_global
        h = torch.cat((latent, h_path, h_global), dim=1)
        if decode_left:
            decoder_states = self.latent2state_l(h).reshape(batch_size, -1, 2) #[bs, 2*hidden_dim, n_layers]
        else:
            decoder_states = self.latent2state_r(h).reshape(batch_size, -1, 2)
        hidden_shape = (
            batch_size, self.decoder.hidden_dim, self.decoder.n_layers
        ) #[bs, hidden_dim, n_layers] = [256, 64, 2]
        if DEBUG:
            print(f"f[SHAPE] model: ConditionalSeq2SeqVAE._get_decoder_states: decoder_states.shape {decoder_states.shape}")
            print(f"f[SHAPE] model: ConditionalSeq2SeqVAE._get_decoder_states: hidden_shape {hidden_shape}")

        hidden = decoder_states[:, :, 0].reshape(*hidden_shape) #256, 64, 2
        hidden = hidden.permute(2, 0, 1).contiguous() #[2, 256, 64] [n_layers, bs, hidden_dim]

        cell = decoder_states[:, :, 1].reshape(*hidden_shape)
        cell = cell.permute(2, 0, 1).contiguous() #[n_layers, bs, hidden_dim]
        if DEBUG:
            print(f"f[SHAPE] model: ConditionalSeq2SeqVAE._get_decoder_states: hidden.shape {hidden.shape}")
            print(f"f[SHAPE] model: ConditionalSeq2SeqVAE._get_decoder_states: cell.shape {cell.shape}")
        return hidden, cell

    def forward(self, prefix, seq_len, window_len, target_l, target_r, target_seq_len, node, offset, edge,
                teacher_force=0.5, need_gauss=False):
        """
        seq_len: [bs, max_widnow_len] = [256, 4]
        prefix: [bs, max_window_len, max_src_length, data_dim] = [256, 4, 32, 3]
        
        all_seq_len: [bs, total_num_prefix_br, ]
        all_seq: [total_num_prefix_br]
        """
        # prefix = [bs, max window len, max seq len, data dim]
        # seq_len = [bs, max window len] -> stores len of each prefix branch per batch branch (max window len no. of prefix branches per branch)
        # target = [bs, max seq len, data dim]; max seq len = L (arg.max_length)
        # target_seq_len = [bs,2]
        # encoder is just a branch encoder
        batch_size, max_wind_l, max_seq_l, input_dim = prefix.shape
        target_l_seq_len = target_seq_len[:, 0] #all left child target branch len in batch
        target_r_seq_len = target_seq_len[:, 1]
        output_dim = self.decoder.output_dim #args.dim = 64

        # get h_path using pooling
        all_seq_len, all_seq = [], []
        # window_len: (batch x 1): each batch dim holds an int of window len
        for idx, t in enumerate(window_len): #for each branch sample, extarct window_len no. of prefix branches
            #t = num of preix branches to consider per branch in a batch dim
            #idx = batch dim (sample id)
            all_seq_len.extend(seq_len[idx][:t]) 
            all_seq.append(prefix[idx][:t])
        if DEBUG:
            print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode:torch.unique(window_len) find all vals in wind_len", torch.unique(window_len)) #elems val <= 4
            print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: seq_len.shape ", seq_len.shape) 
            print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: prefix.shape ", prefix.shape) 
            # print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: all_seq_len ", all_seq_len) 
            # print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: all_seq ", all_seq) 
          
        #all_seq_len = [batch * window_len]: 1D list of all branch lengths for prefix branches across batch
        #before permute:  all_seq =  [batch*t, seq_len, 3] all prefix branches 
        # all_seq = [max seq len, sum(window_len), data dim] after permute
            #sum(window_len) = sum of all the window_len in the batch i.e. batch*t assuming uniform window_len
            # max seq len = seq_len 
            #all_seq[i,:,:] gives the 3D coordinates of all the ith point on each prefix branch in the batch

        all_seq = torch.cat(all_seq, dim=0).permute(1, 0, 2) # [max_src_length, total_prefix_no, data_dim] = [32, *, 3]
        if DEBUG:
            print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: all_seq.shape ", all_seq.shape) 

        #all_seq_len: [total_prefix_no]; all_seq: [max_src_length, total_prefix_no, data_dim] 
        hiddens, cells = self.encoder(all_seq, all_seq_len) 
        #hiddens, cells: [no.layers, total_prefix_no, hidden_dim] = [2, * 64]
        if DEBUG:
            print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: hiddens.shape", hiddens.shape)
            print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: cells.shape", cells.shape)

        #encoded rep for every prefix branches (of each sample in the batch)
        hidden_seq = torch.cat([hiddens, cells], dim=-1) #[no.layers, total_prefix_no, hidden_dim*2] = [2, *, 128]
        if DEBUG:
            print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: hidden_seq.shape", hidden_seq.shape)
        
        #h_local = h_path encoding 
        # pooling to get h_path = [bs, n_layers*2*hidden_dim]
        if self.remove_path:
            h_path = torch.zeros([batch_size, hidden_seq.shape[0] * hidden_seq.shape[2]]).to(self.device)
        else:
            if self.new_model:
                #use lstm to aggregate r_b reps
                # cond_hidden or cond_cell: [n_layers, batch_size, hidden_dim]
                cond_hidden, cond_cell = self.condition_encoder(prefix, seq_len, window_len)
                #  h_path = [2*n_layers, batch_size, hidden_dim] aka [[hidden 1 for all prefix branches], [hidd 2], [cell1 ], [cell2]]
                h_path = torch.cat([cond_hidden, cond_cell], dim=0)
                # after permute h_path = [batch_size,2*n_layers, hidden_dim] aka [[[hidd 1 for b1's prefix branches], [hidd 2 for b1's], [cell1 for b1's], [cell2 for b1's]], [state encodings for 2nd branch's prefix branches in batch ], ...]
                # after reshape (flatten): [batch_size, 2*n_layers*hidden_dim] = [batch_size, 256] aka [[embedding rep for b1], [embedding rep for the prefix path to ending at 2nd branch in batch]] 
                h_path = h_path.permute(1, 0, 2).reshape(batch_size, -1)
            else:
                h_path = []
                pre_cnt = 0 # controls which set of prefixes corresponds to which sample
                #use EMA (method in paper)
                def compress_path_embedding(y, forgettable, device):
                    if forgettable == None: #mean pooling 
                        return torch.mean(y, 1) #single h_path: [no_layers, hidden_dim*2] = [2, 128]
                    else: #EMA eqn 
                        h = torch.zeros(y.shape[0], y.shape[2]).to(device)
                        for i in range(y.shape[1]): #y.shape[1] deals with a branch 
                            h = forgettable * h + (1 - forgettable) * y[:, i, :]
                        return h

                for _, wl in enumerate(window_len): #loop ensures final len(h_path) = bs
                    h_path.append(
                        compress_path_embedding(hidden_seq[:, pre_cnt:pre_cnt + wl, :], self.forgettable, self.device))
                    pre_cnt += wl
                #h_path: [bs, no_layers * hidden_dim*2] = [bs, 2 * hidden_dim*2] = [256, 256]
                h_path = torch.stack(h_path).reshape(batch_size, -1)
                if DEBUG:
                    print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: h_path.shape:", h_path.shape)
        # h_global encoding
        if self.remove_global:
            h_global = torch.zeros([batch_size, self.global_dim]).to(self.device)
        else:
            #offset: idx of data sample the corresponding prev branch in node belongs to 
            if offset.shape[0] == 0:
                #empty tree
                h_global = torch.zeros([batch_size, self.global_dim]).to(self.device)
            else:
                #branch encoding for previous branches (nodes)
                # node (before permute): [total_nodes, L, 3], total_nodes = total_no_prev_br
                node = node.permute(1, 0, 2) #[L, total_nodes, 3]
                #assume every branch is the same length, there are node.shape[1] number of branches in the batch
                node_len = target_l_seq_len[0] * torch.ones(node.shape[1]) # [total_nodes] i.e. a list of prev branch lens
                #hidden, cell: [n_layers, total_nodes, hidden_dim] = [2, *, 64]
                hidden, cell = self.encoder(node, node_len)
                node = torch.cat((hidden, cell), dim=0) # [2*n_layers, total_nodes, hidden_dim] = [4, *, 64]
                if DEBUG:
                    print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode h_global: hidden.shape:", hidden.shape)
                    print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: h_global: cell.shape:", cell.shape)
                    print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: h_global: node_len.shape:", node_len.shape)
                    
                #branch encoding for each node after permute: [total_nodes, 2 * n_layers, hidden_dim]
                node = node.permute(1, 0, 2).reshape(hidden.shape[1], -1) # [total_nodes, 2*n_layers*hidden_dim] = [*, 256]
                if DEBUG:
                    print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: h_global: node.shape:", node.shape)
                    print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: h_global: edge.shape:", edge.shape)
                # node: [total_nodes, 2*n_layers*hidden] = [total_nodes, 256]
                #offset: [total_nodes]: map node to sample in batch
                # edge: [_max+1, total_nodes, total_nodes] (sparse): max = max depth of any branch in node
                h_global = self.tgnn(node, offset, edge)
                if DEBUG:
                    print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: h_global: h_global.shape:", h_global.shape)
                    print("[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: h_global : self.global_dim", self.global_dim)

        target_l = target_l.permute(1, 0, 2) #move batch dim to second dim, group by 1st coord etc. 
        target_r = target_r.permute(1, 0, 2)
        if need_gauss:
            h = 0
            Z = self.gauss.sample().to(self.device)
            Z = Z / torch.norm(Z)
        else:
            #VAE encoder logic 
            h, Z, condition = self.encode(target_l, target_l_seq_len, target_r, target_r_seq_len, h_path, h_global)
        hidden, cell = self._get_decoder_states(Z, batch_size, h_path, h_global, True)

        target_len = target_l.shape[0]

        output_l = conditional_decode_seq(
            self.decoder, (target_len, batch_size, output_dim),
            target_l[0], hidden, cell, self.device,
            teacher_force=teacher_force, target=target_l
        ) #[max_dst_len, bs, data_dim] = [32, 256, 3]
        if DEBUG: print(f"[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: output_l.shape: {output_l.shape}")
        output_l = output_l.permute(1, 0, 2) #move bs to dim0, [bs, max_dst_len, data_dim]
        if DEBUG: print(f"[DEBUG] model.py: ConditionalSeq2SeqVAE.encode: output_l.permute(1, 0, 2).shape: {output_l.shape}")

        hidden, cell = self._get_decoder_states(Z, batch_size, h_path, h_global, False)

        target_len = target_r.shape[0]

        output_r = conditional_decode_seq(
            self.decoder, (target_len, batch_size, output_dim),
            target_r[0], hidden, cell, self.device,
            teacher_force=teacher_force, target=target_r
        )
        output_r = output_r.permute(1, 0, 2)
        return output_l, output_r, h, Z




class BranchEncRnn(torch.nn.Module):
    def __init__(
        self, input_dim, embedding_dim, hidden_dim,
        n_layers=2, dropout=0.5
    ):
        super(BranchEncRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.coordinate2emb = torch.nn.Linear(input_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0
        )

    def forward(self, src, seq_len, return_in_one=False):
        input_seq = self.dropout_fun(self.coordinate2emb(src))
        packed_seq = pack_padded_sequence(
            input_seq, seq_len, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.rnn(packed_seq)

        # outputs = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        #
        if not return_in_one:
            return hidden, cell
        else:
            batch_size = hidden.shape[1]
            answer = torch.cat([hidden, cell], dim=0)
            answer = answer.permute(1, 0, 2).reshape(batch_size, -1)
            return answer


class BranchDecRnn(torch.nn.Module):
    def __init__(
        self, output_dim, embedding_dim, hidden_dim,
        n_layers=2, dropout=0.5
    ):
        super(BranchDecRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hid2coordinate = torch.nn.Linear(hidden_dim, output_dim)
        self.coordinate2emb = torch.nn.Linear(output_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0
        )

    def decode_a_step(self, init, hidden, cell):
        init = init.unsqueeze(0)
        # init = [1, batch_size, output_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        embedding = self.dropout_fun(self.coordinate2emb(init))
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        prediction = self.hid2coordinate(output).squeeze(0)
        return prediction, hidden, cell

    def forward(
        self, hidden, cell, target_len=None, target=None,
        teaching=0.5, init=None
    ):
        if target_len is None and target is None:
            raise ValueError('the target_length should be specified')
        if init is None and target is None:
            raise ValueError('the start point should be specified')
        if teaching > 0 and target is None:
            raise NotImplementedError(
                'require stadard sequence as input'
                'when using teacher force'
            )

        if target_len is None:
            target_len = target.shape[0]
        if init is None:
            init = target[0]

        batch_size = hidden.shape[1]
        output_shape = (target_len, batch_size, self.output_dim)
        outputs = torch.zeros(output_shape).to(hidden.device)
        outputs[0] = init
        current = outputs[0].clone()
        for t in range(1, target_len):
            output, hidden, cell = self.decode_a_step(current, hidden, cell)
            outputs[t] = output
            current = target[t] if random.random() < teaching else output
        return outputs




class RNNAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(RNNAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, src, seq_len, target=None, teaching=0.5,
        init=None, target_len=None
    ):
        hidden, cell = self.encoder(src, seq_len, return_in_one=False)
        return self.decoder(
            hidden, cell, target_len=target_len,
            target=target, init=init, teaching=teaching
        )