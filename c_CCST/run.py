from .CCST import train_DGI, PCA_process, get_graph, Kmeans_cluster
from torch_geometric.data import DataLoader
import torch
import numpy as np
from torch_geometric.data import Data
import scanpy as sc


def adata_preprocess(i_adata, min_cells=3, pca_n_comps=300):
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X


def get_adj(coordinates, threshold_list=[300]):
    # coordinates = np.load(generated_data_fold + 'coordinates.npy')
    # if not os.path.exists(generated_data_fold):
    #     os.makedirs(generated_data_fold) 
    ############# get batch adjacent matrix
    cell_num = len(coordinates)

    ############ the distribution of distance 
    # if 1:#not os.path.exists(generated_data_fold + 'distance_array.npy'):
    distance_list = []
    print ('calculating distance matrix, it takes a while')
    
    for j in range(cell_num):
        for i in range (cell_num):
            if i!=j:
                distance_list.append(np.linalg.norm(coordinates[j]-coordinates[i]))

    distance_array = np.array(distance_list)
    #np.save(generated_data_fold + 'distance_array.npy', distance_array)
    # else:
    #     distance_array = np.load(generated_data_fold + 'distance_array.npy')

    ###try different distance threshold, so that on average, each cell has x neighbor cells, see Tab. S1 for results
    from scipy import sparse
    import pickle
    import scipy.linalg

    for threshold in threshold_list:#range (210,211):#(100,400,40):
        num_big = np.where(distance_array<threshold)[0].shape[0]
        print (threshold,num_big,str(num_big/(cell_num*2))) #300 22064 2.9046866771985256
        from sklearn.metrics.pairwise import euclidean_distances

        distance_matrix = euclidean_distances(coordinates, coordinates)
        distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
        distance_matrix_threshold_W = np.zeros(distance_matrix.shape)
        for i in range(distance_matrix_threshold_I.shape[0]):
            for j in range(distance_matrix_threshold_I.shape[1]):
                if distance_matrix[i,j] <= threshold and distance_matrix[i,j] > 0:
                    distance_matrix_threshold_I[i,j] = 1
                    distance_matrix_threshold_W[i,j] = distance_matrix[i,j]
            
        
        ############### get normalized sparse adjacent matrix
        distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I) ## do not normalize adjcent matrix
        return sparse.csr_matrix(distance_matrix_threshold_I_N)
        # with open(generated_data_fold + 'Adjacent', 'wb') as fp:
        #     pickle.dump(distance_matrix_threshold_I_N_crs, fp)


def train(adata, radius, n_cluster, epochs=5000, seed=2022, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), embedding_size=256, lr=1e-6):
    features = adata_preprocess(adata)
    adata.obsm['adj'] = get_adj(adata.obsm['spatial'], threshold_list=[radius])
    data = get_graph(adata.obsm['adj'], features)
    data_loader = DataLoader(data, batch_size=1)
    print('>>> graph contains {} edges, {} edges per node'.format(data[0].edge_index.shape[1], data[0].edge_index.shape[1] / data[0].x.shape[0]))
    print('>>> begin to train DGI, shape: ({}, {})'.format(data[0].x.shape[0], data[0].x.shape[1]))
    adata.obsm['CCST'] = train_DGI(data_loader, len(data[0].x[0]), embedding_size, epochs, lr, seed, device)
    adata.obsm['CCST_pca'] = PCA_process(adata.obsm['CCST'], 50)
    adata.obs['CCST'], _ = Kmeans_cluster(adata.obsm['CCST'], n_cluster)
    adata.obs['CCST_pca'], _ = Kmeans_cluster(adata.obsm['CCST_pca'], n_cluster)

