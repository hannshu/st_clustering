# 目录
- [1 空间转录组数据聚类分析方法](#整理到的对空间转录组进行聚类分析的算法)  
- [2 空间转录组数据集](#目前整理到可以使用的数据集)  
- [3 聚类评价方法](#评价指标)

# 整理到的对空间转录组进行聚类分析的算法

|    | model | published time | method | journal | paper | status | tutorial | link | note  
| -- | ----- | -------------- | ------ | ------- | ----- | ------ | -------- | ---- | ---- 
|1|Louvain|2008.7|community discovery|Journal of Statistical Mechanics: Theory and Experiment|[DOI](https://doi.org/10.1088/1742-5468/2008/10/P10008)|*|[tutorial](./1_ScanPy/train.ipynb)|[link](#louvain)|  
|2|STAGATE|2022.4|autoencoder+GAT|Nature Communications|[DOI](https://doi.org/10.1038/s41467-022-29439-6)|*|[tutorial](./2_STAGATE_pyG/train.ipynb)|[link](#stagate)|只有tf实现中可以使用alpha，torch实现中没有这个功能  
|3|CCST|2022.6|DGI(GCN)|Nature Computational Science|[DOI](https://doi.org/10.1038/s43588-022-00266-5)|*|[tutorial](./3_CCST/train.ipynb)|[link](#ccst)|  
|4|SpaGCN|2021.10|GCN|Nature Methods|[DOI](https://doi.org/10.1038/s41592-021-01255-8)|*|[tutorial](./4_SpaGCN/train.ipynb)|[link](#spagcn)|算法中用到的组织学数据需要自行制作  
|5|SEDR|||Research Square|[DOI](https://doi.org/10.21203/rs.3.rs-665505/v1)|NULL|[tutorial]()|[link](#sedr)|  
|6|BayesSpace|2021.6|static|Nature Biotechnology|[DOI](https://doi.org/10.1038/s41587-021-00935-2)|NULL|[tutorial]()|[link](#bayesspace)|  
|7|stLearn|||bioRxiv|[DOI](https://doi.org/10.1101/2020.05.31.125658)|NULL|[tutorial]()|[link](#stlearn)|  
|8|Giotto|||BMC Genome Biology|[DOI](https://doi.org/10.1186/s13059-021-02286-2)|NULL|[tutorial]()|[link](#giotto)|  
|9|TransformerST|||bioRxiv|[DOI](https://doi.org/10.1101/2022.08.11.503261)|NULL|[tutorial]()|[link](#transformerst)|  
|10|FICT|2021.10|static(EM)|Bioinformatics|[DOI](https://doi.org/10.1093/bioinformatics/btab704)|NULL|[tutorial]()|[link](#fict)|  

## louvain, STAGATE, CCST, SpaGCN  
详见: [link](https://blog.hanshu.org/%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84/2022-11-01-louvain,%20STAGATE,%20CCST,%20SpaGCN%E6%80%BB%E7%BB%93/)  


## DeepST  
详见: [link](https://blog.hanshu.org/%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84/2022-10-25-DeepST:-identifying-spatial-domains-in-spatial-transcriptomics-by-deep-learning/)    
VAE详见: [link](https://blog.hanshu.org/ae/2022-10-25-%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E4%BB%8B%E7%BB%8D/)  

## SEDR



## BayesSpace



## stLearn



## Giotto



## TransformerST



## FICT
  


# 目前整理到可以使用的数据集

|    | name | source | paper | annotation | size  (cell/spot * gene) | note  
| -- | ---- | ------ | ----- | ---------- | ------------------------ | ----
|1|MERFISH|squidpy|[DOI](https://doi.org/10.1126/science.aau5324)|*|73655*161|12 slices  
|2|MIBI-TOF|squidpy|[DOI](https://doi.org/10.1101/2020.01.17.909796)|*|3309*36|3 slices  
|3|SlideseqV2|squidpy|[DOI](https://doi.org/10.1038/s41587-020-0739-1)|*|41786*4000|mouse neocortex  
|4|scRNA-seq mouse cortex|squidpy|[DOI](https://doi.org/10.1038/s41586-018-0654-5)|*|21697*36826|simulated  
|5|10x Visium (DLPFC dataset)|spatialLIBD|[DOI](https://doi.org/10.1186/s12864-022-08601-w)|*|approx. 4000*33538 each|12 slices  
|6|10x Genomics Adult Mouse Brain Section 1 (Coronal)|10x Genomics|[10x Genomics](https://www.10xgenomics.com/resources/datasets/adult-mouse-brain-section-1-coronal-stains-dapi-anti-neu-n-1-standard-1-1-0)|-|2903*32285|  
|7|Mouse Brain Serial Section 1 (Sagittal-Posterior)|10x Genomics|[10x Genomics](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.0.0/V1_Mouse_Brain_Sagittal_Posterior)|-|3353*31053|
|8|Slide-seqV2|BROAD INSTITUTE|[BROAD INSTITUTE](https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary)|-|21724*21220|mouse olfactory bulb 
|9|10x Genomics Visium H&E dataset|squidpy(10x Genomics)|[10x Genomics](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Adult_Mouse_Brain)|*|2688*18078|coronal section of the mouse brain
|10|10x Genomics Visium Fluorecent dataset|squidpy(10x Genomics)|[10x Genomics](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Adult_Mouse_Brain_Coronal_Section_2)|*|2800*16562|

## squidpy pre-processed
所有经过squidpy预处理的数据集均附带有手工标记，文件格式为```.h5ad```，位置信息保存在```adata.obsm['spatial']```中，手工标记信息一般保存在```adata.obs['Cluster']```中，squidpy中集成了内置的读取方法来读取数据集  
- [MERFISH](https://ndownloader.figshare.com/files/28169379):
```adata = sq.datasets.merfish(path=os.path.join('dataset', 'merfish3d.h5ad'))```  
每个slice保存在```adata.obs['Bregma']```中，分别是```{-28.999999999999996, -24.0, -19.0, -14.000000000000002, -9.0, -4.0, 1.0, 6.0, 11.0, 16.0, 21.0, 26.0}```，手工标记的类别保存在```adata.obs['Cell_class']```中  
- [MIBI-TOF](https://ndownloader.figshare.com/files/28241139):  
```adata = sq.datasets.mibitof(path=os.path.join('dataset', 'ionpath.h5ad'))```  
每个slice保存在```adata.obs['batch']```中，分别是```{'0', '1', '2'}```  
- [SlideseqV2](https://ndownloader.figshare.com/files/28242783): ```adata = sq.datasets.slideseqv2(path=os.path.join('dataset', 'slideseqv2.h5ad'))```
- [scRNA-seq mouse cortex](https://ndownloader.figshare.com/files/26404781): ```adata = sq.datasets.sc_mouse_cortex(path=os.path.join('dataset', 'sc_mouse_cortex.h5ad'))```
- [10x Genomics Visium H&E dataset](https://ndownloader.figshare.com/files/26098397):  ```adata = sq.datasets.visium_hne_adata(path=os.path.join('dataset', 'visium_hne.h5ad'))```
- [10x Genomics Visium Fluorecent dataset](https://ndownloader.figshare.com/files/26098391): ```adata = sq.datasets.visium_fluo_adata(path=os.path.join('dataset', 'visium_fluo.h5ad'))```

## DLPFC数据集
DLPFC数据集是经过预处理的数据集，并包含有手工标注信息，文件格式为```.h5```，scanpy中集成有读取这种数据集的函数
``` python
section_list = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']

# 读取section_id这个slice的数据
adata = sc.read_visium(path=os.path.join('dataset', 'DLPFC', section_id))
adata.var_names_make_unique()

# 为数据添加ground truth
Ann_df = pd.read_csv(os.path.join('dataset', 'DLPFC', section_id, 'ground_truth.txt'), sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
adata.obs['Cluster'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
```

## 10x Genomic
``` python
# Coronal:
adata = sc.read_visium(path=os.path.join('dataset', 'Adult_Mouse_Brain', 'Coronal'))
adata.var_names_make_unique()

# Sagittal-Posterior:
adata = sc.read_visium(path=os.path.join('dataset', 'Adult_Mouse_Brain', 'Sagittal-Posterior'))
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

# 评价指标
[参考链接](https://zhuanlan.zhihu.com/p/343667804)
## 兰德系数
TP：表示两个同类样本点在同一个簇中的情况数量(从一个*簇*中抽取两个，是同一*类*的个数)   
FP：表示两个非同类样本点在同一个簇中的情况数量(从一个*簇*中抽取两个，不是同一个*类*的个数)  
TN：表示两个非同类样本点分别在两个簇中的情况数量(任意两个*簇*中抽取两个，这两个不是同一个*类*)  
FN：表示两个同类样本点分别在两个簇中的情况数量(任意两个*簇*中抽取两个，这两个是同一个*类*)  
**注**: 这里的*簇*表示这个spot的预测值，*类*表示spot的真值  
则兰德系数的计算公式如下:  
$$
RI = \frac{TP + TN}{TP + FP + TN + FN}
$$

## F值
$$
Precision = \frac{TP}{TP + FP} \\ 
Recall = \frac{TP}{TP + FN} \\ 
F_{\beta} = (1 + \beta^2) \frac{Precision * Recall}{\beta^2 * Precision + Recall}
$$

这里可以发现，TP是可以直接计算得到: $TP = \sum^{簇个数} \sum^{每个簇中个数大于2的类} C^{这个类在簇中的个数}_2$  
其他三个参数不能通过计算直接得到，但可以得到一些参数的和:  
其中
$$
TP + FN = \sum^{类} C^{这个类在所有簇中的总个数}_2 \\
TP + FP = \sum^{簇} C^{簇中元素的个数}_2 \\
TP + FP + TN + FN = C^{总共有多少个数}_2
$$

## 调整兰德系数
$$
ARI = \frac{2 * (TP * TN  - FN * FP)}{(TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)}
$$
ARI的取值范围是[-1, 1]，RI的取值范围是[0, 1]  

## 聚类纯度
$$
P = \frac{1}{N} \sum_k \max_j |\omega_k \cap c_j|
$$
其中$\omega_k$为聚类后的第k个类，$c_j$为这个元素属于第j类  
每次选取每个簇中包含最多的那个元素的元素个数，然后求和取平均  