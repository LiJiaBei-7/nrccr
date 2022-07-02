import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import math
import copy



class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class BertSelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        super(BertSelfAttention, self).__init__()
        self.attention_probs_dropout_prob = 0.1
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size    # =hidden_size
        self.key_query_size = 512
        self.query = nn.Linear(hidden_size, self.key_query_size)
        self.key = nn.Linear(hidden_size, self.key_query_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, int(x.size(-1)/self.num_attention_heads))  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def get_attention_mask(self, mask_query, mask):
        attention_mask = torch.matmul(mask_query.transpose(-1, -2), mask)
        return attention_mask

    def forward(self, query_states, key_states, value_states, attention_mask_query, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = self.get_attention_mask(attention_mask_query, attention_mask)
        attention_mask = (1.0 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)

        mixed_query_layer = self.query(query_states)    # (N, L, d)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_mask = attention_mask.expand(attention_mask.size(0), self.num_attention_heads, attention_mask.size(-2),-1)
        # print(attention_scores[0][0].min(-1))
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer



class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, ):
        super(BertSelfOutput, self).__init__()
        self.hidden_dropout_prob = 0.1
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, cross):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if cross:
            hidden_states = self.LayerNorm(hidden_states)
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertAttention(nn.Module):
    def __init__(self, opt, num_attention_heads, hidden_size):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(num_attention_heads, hidden_size)
        self.output = BertSelfOutput(hidden_size)

    def forward(self, input_tensor_query, input_tensor, attention_mask_query, attention_mask, cross=False):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor_query, input_tensor, input_tensor, attention_mask_query, attention_mask)
        attention_output = self.output(self_output, input_tensor_query, cross)
        return attention_output
