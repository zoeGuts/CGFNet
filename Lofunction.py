import torch


def l2_norm(x, axis=1):
    ''' l2 norm

    '''
    norm = torch.norm(x, 2, axis, True)
    outs = torch.div(x, norm)
    
    return outs


def Focal_Attention_Loss(attn, y, sigma=0.01):
    '''
        attn: [b, c, 1, 1]
        y :[b]
    '''
    B, C, _, _ = attn.shape
    attn = attn.squeeze(-1).squeeze(-1)  # [b, c]
    matrix_E = torch.sum(attn * attn, dim=1).unsqueeze(-1).repeat(1, B)   # [b, b]
    matrix_G = attn @ attn.T
    
    matrix_D = matrix_E + matrix_E.T - 2 * matrix_G
    similarity_mask = torch.stack([y == y[i] for i in range(B)]).float()  # mask --> same class    
    
    intra_L = torch.sum(torch.mean(matrix_D * similarity_mask, dim=1))
    inter_L = torch.sum(torch.mean(matrix_D * (1 - similarity_mask), dim=1))
    
    total_L = intra_L / (inter_L + sigma)
    
    return total_L
