# 整理到的对空间转录组进行聚类分析的算法

|    | model | journal | paper | status | tutorial | note  
| -- | ----- | ------- | ----- | ------ | -------- | ---- 
|1|Leiden|Scientific Reports|[DOI](https://doi.org/10.1038/s41598-019-41695-z)|NULL|[tutorial]()|  
|2|STAGATE|Nature Communications|[DOI](https://doi.org/10.1038/s41467-022-29439-6)|*|[tutorial](./2_STAGATE_pyG/train.ipynb)|只有tf实现中可以使用alpha，torch实现中没有这个功能  
|3|CCST|Nature Computational Science|[DOI](https://doi.org/10.1038/s43588-022-00266-5)|NULL|[tutorial]()|  
|4|SpaGCN|Nature Methods|[DOI](https://doi.org/10.1038/s41592-021-01255-8)|NULL|[tutorial]()|  
|5|SEDR|-|[DOI](https://doi.org/10.21203/rs.3.rs-665505/v1)|NULL|[tutorial]()|  
|6|BayesSpace|Nature Biotechnology|[DOI](https://doi.org/10.1038/s41587-021-00935-2)|NULL|[tutorial]()|  
|7|stLearn|bioRxiv|[DOI](https://doi.org/10.1101/2020.05.31.125658)|NULL|[tutorial]()|  
|8|Giotto|BMC Genome Biology|[DOI](https://doi.org/10.1186/s13059-021-02286-2)|NULL|[tutorial]()|  
|9|TransformerST|bioRxiv|[DOI](https://doi.org/10.1101/2022.08.11.503261)|NULL|[tutorial]()|  
|10|FICT|Bioinformatics|[DOI](https://doi.org/10.1093/bioinformatics/btab704)|NULL|[tutorial]()|  
|11|scGNN|Nature Communications|[DOI](https://doi.org/10.1038/s41467-021-22197-x)|NULL|[tutorial]()|  

## Leiden


## STAGATE
![img](./2_STAGATE_pyG/STAGATE_Overview.png)  
使用类似于自编码器的结构，在聚合邻边信息时引入attention  

- 生成图:  
使用KNN或是给定半径  
在生成图时可以先进行一次pre-cluster，切断聚类中不同类之间的边  

- GNN:  
如结构图中所示，使用一个类似自编码器的结构  
在编码的过程中:  
对于每次聚合 $h^{(k)}_i = \sigma(\sum_{j\in S_i} att^{(k)}_{ij} (W_kh^{(k-1)}_j))$  
最后一层全连接 $h^{(k)}_i = \sigma(W_kh^{(k-1)}_j)$  
解码过程和编码过程类似，不同的是解码时使用的att值和编码时相同，解码时使用的全连接层是编码时使用的全连接层的转置   
attention机制:  
训练邻边和自身的向量$v_s$和$v_r$  
$$
e^{(k)}_{ij} = \sigma(v^{(k)^T}_s (W_kh^{k-1}_i) + v^{(k)^T}_r (W_kh^{k-1}_j)), \\
att^{(k)}_{ij} = \frac{exp(e^{(k)}_{ij})}{\sum_{i \in N(i)}e^{(k)}_{ij}},
$$
如果使用到了Construction of cell type-aware SNN，则只需要按设定好的$\alpha$将两个attention相加即可
$$
att_{ij} = (1 - \alpha)att^{spatial}_{ij} + \alpha att^{aware}_{ij}
$$

- loss:  
主要思想是经过编码/解码后得到的值应该和原值类似，所以loss函数设置为前后两向量的距离$\sum^N_{i=1} || x_i - \hat{h}^0_i ||_2$

## CCST
![img](./3_CCST/figure1.png)  
使用DGI模型对空间转录组数据进行处理    

- 生成图:  
给定超参数$d_{thres}$，计算每两个节点间的距离，如果小于超参数，则生成边，否则不生成边。  
为了平衡连接边的权重和节点的基因表达信息，引入超参数$\lambda$，修正邻接矩阵$A = \lambda * I + (1 - \lambda) * A_0$  

- 数据预处理:   
对于每个数据集，需要删除表达量较低的基因，对每个spot的基因表达进行归一化。

- GNN:  
采用DGI模型，loss函数：
$$Loss = \sum^N_{i = 1} E_{X, A}[\log D(h_i, s)] + E_{X, \overline{A}}[\log D(\overline{h_i}, s)]$$ 
其中:  
X为每个细胞的基因表达矩阵。  
A为正确图的邻接矩阵。  
$\overline{A}$为生成的混淆图的邻接矩阵(这里混淆图的边是随机生成的)。  
$h_i$是正常图经过GCN后得到的embedding vector。  
$\overline{h_i}$是混淆图经过同一个GCN后得到的embedding vector。  
s是正常图经过GCN后得到所有embedding vector的平均值(用来代表整张图)。  
DGI的主要思想是*最大化***混淆图生成的向量**和图向量间的距离，同时*最小化***正常图生成的向量**和图向量之间的距离。  

- 聚类:  
得到embedding vector后先通过PCA进行降维，然后使用umap进行聚类和可视化。  

- Differential gene expression analysis:  
秩和检验

## SpaGCN



## SEDR



## BayesSpace



## stLearn



## Giotto



## TransformerST



## FICT
  


## scGNN


# 目前整理到可以使用的数据集

|    | name | source | paper | annotation | size  (cell/spot * gene) | note  
| -- | ---- | ------ | ----- | ---------- | ------------------------ | ----
|1|MERFISH|squidpy|[DOI](https://doi.org/10.1126/science.aau5324)|*|73655*161|  
|2|MIBI-TOF|squidpy|[DOI](https://doi.org/10.1101/2020.01.17.909796)|*|3309*36|3 slices  
|3|SlideseqV2|squidpy|[DOI](https://doi.org/10.1038/s41587-020-0739-1)|*|41786*4000|mouse neocortex  
|4|scRNA-seq mouse cortex|squidpy|[DOI](https://doi.org/10.1038/s41586-018-0654-5)|*|21697*36826|  
|5|10x Visium (DLPFC dataset)|spatialLIBD|[DOI](https://doi.org/10.1186/s12864-022-08601-w)|*|approx. 3460*33538 each|12 slices  
|6|10x Genomics Adult Mouse Brain Section 1 (Coronal)|10x Genomics|[10x Genomics](https://www.10xgenomics.com/resources/datasets/adult-mouse-brain-section-1-coronal-stains-dapi-anti-neu-n-1-standard-1-1-0)|-|2903*32285|  
|7|Slide-seqV2|BROAD INSTITUTE|[BROAD INSTITUTE](https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary)|-|21724*21220|mouse olfactory bulb  

## squidpy pre-processed
所有经过squidpy预处理的数据集均附带有手工标记，文件格式为```.h5ad```，一般保存在```adata.obsm['spatial']```中，squidpy中集成了内置的读取方法来读取数据集  
- [MERFISH](https://ndownloader.figshare.com/files/28169379): ```adata = sq.datasets.merfish(path=os.path.join('dataset', 'merfish3d.h5ad'))```
- [MIBI-TOF](https://ndownloader.figshare.com/files/28241139): ```adata = sq.datasets.mibitof(path=os.path.join('dataset', 'ionpath.h5ad'))```
- [SlideseqV2](https://ndownloader.figshare.com/files/28242783): ```adata = sq.datasets.slideseqv2(path=os.path.join('dataset', 'slideseqv2.h5ad'))```
- [scRNA-seq mouse cortex](https://ndownloader.figshare.com/files/26404781): ```adata = sq.datasets.sc_mouse_cortex(path=os.path.join('dataset', 'sc_mouse_cortex.h5ad'))```

## DLPFC数据集
DLPFC数据集是经过预处理的数据集，并包含有手工标注信息，文件格式为```.h5```，scanpy中集成有读取这种数据集的函数
``` python
# 读取section_id这个slice的数据
adata = sc.read_visium(path=os.path.join('dataset', 'DLPFC', section_id))
adata.var_names_make_unique()

# 为数据添加ground truth
Ann_df = pd.read_csv(os.path.join('dataset', 'DLPFC', section_id, 'ground_truth.txt'), sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
```

## 10x Visium Mouse Brain Section 1
``` python
adata = sc.read_visium(path=os.path.join('dataset', 'Adult_Mouse_Brain_Section_1'))
adata.var_names_make_unique()
```

## Slide-seqV2
这个数据集比较特殊，是通过读取csv和txt文件得到spot的属性信息和位置信息，然后再自己建立AnnData类
``` python
# 读取文件数据，其中counts保存了spot的features，coor_df保存了spot的位置信息
counts = pd.read_csv(os.path.join('dataset', 'Slide-seqV2_MoB', 'data', 'Puck_200127_15.digital_expression.txt'), sep='\t', index_col=0)
coor_df = pd.read_csv(os.path.join('dataset', 'Slide-seqV2_MoB', 'data', 'Puck_200127_15_bead_locations.csv'), index_col=0)

# 生成AnnData
adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
adata.obsm["spatial"] = coor_df.to_numpy()

# 删除掉表达过低的spot
used_barcode = pd.read_csv(os.path.join('dataset', 'Slide-seqV2_MoB', 'used_barcodes.txt'), sep='\t', header=None)
adata = adata[used_barcode[0],]
```