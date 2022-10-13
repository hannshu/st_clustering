from .Train_STAGATE import train_STAGATE
from .utils import Cal_Spatial_Net
import torch


def train(adata, epochs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), k_cutoff=None, rad_cutoff=None, seed=2022):
    if (k_cutoff):
        model = 'KNN'
    else:
        model = 'Radius'
    Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff, k_cutoff=k_cutoff, model=model) # 生成网络，保存在.uns['Spatial_Net']中 
    
    return train_STAGATE(adata, n_epochs=epochs, random_seed=seed, device=device)
