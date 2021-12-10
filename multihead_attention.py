from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dim_v=None, project_dim=None, dropout=0., bias=False, rescale_attention=False):
        super(MultiHeadAttention, self).__init__()
        dim_v = dim if dim_v is None else dim_v
        self.dim = dim
        self.dim_v = dim_v
        self.heads = heads
        self.attention_head_size = dim // heads
        self.context_head_size = dim_v // heads
        self.attend = nn.Softmax(dim=-1)
        self.attention_scale = self.attention_head_size ** (-0.5)
        self.to_out = nn.Linear(dim_v, project_dim, bias=bias) if project_dim is not None else nn.Identity()
        self.dropout_attention = nn.Dropout(dropout)
        # TODO(moaba) Remove
        self.rescale_attention = rescale_attention

    def transpose_for_scores(self, x, head_dim_size):
        new_x_shape = x.size()[:-1] + (self.heads, head_dim_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None, return_attention=False, relative_pos=None):
        """
        :param Q: B N D1
        :param K: B N D1
        :param V: B N D2
        :param mask: mask to applied on the attention
        :param return_attention: wether to return the attention or not
        :return: softmax(Q*K.T)*V and softmax(Q*K.T) if return mask is set
        """
        Qh = self.transpose_for_scores(Q, self.attention_head_size)
        Kh = self.transpose_for_scores(K, self.attention_head_size)
        Vh = self.transpose_for_scores(V, self.context_head_size)
        # Calculate attention
        attention_scores = torch.matmul(Qh, Kh.transpose(-1, -2))
        if mask is not None:
            attention_scores -= mask
        if relative_pos is not None:
            attention_scores += relative_pos
        attention_probs = self.dropout_attention(self.attend(attention_scores * self.attention_scale))
        # Calculate output
        context_layer = torch.matmul(attention_probs, Vh)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.context_head_size * self.heads,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.to_out(context_layer)

        return (context_layer, attention_probs) if return_attention else context_layer
