# 整理到的对空间转录组进行聚类分析的算法

## Leiden
[Leiden(ScanPy)](https://doi.org/10.1038/s41598-019-41695-z)   


## STAGATE
[STAGATE](https://doi.org/10.1038/s41598-019-41695-z)    
[tutorial](./STAGATE_pyG/train.ipynb)  
![img](./STAGATE_pyG/STAGATE_Overview.png)  
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
主要思想是经过编码/解码后得到的值应该和原值类似，所以loss函数设置为前后两向量的距离  
$\sum^N_{i=1} || x_i - \hat{h}^0_i ||_2$

备注: 只有tf实现中可以使用alpha，torch实现中没有这个功能


## CCST
[CCST](https://doi.org/10.21203/rs.3.rs-990495/v1)  


## SpaGCN
[SpaGCN](https://doi.org/10.21203/rs.3.rs-990495/v1)  


## SEDR
[SEDR](https://doi.org/10.21203/rs.3.rs-665505/v1)  


## BayesSpace
[BayesSpace](https://doi.org/10.1101/2020.05.31.125658)  


## stLearn
[stLearn](https://doi.org/10.1101/2020.05.31.125658)  


## Giotto
[Giotto](https://doi.org/10.21203/rs.3.rs-990495/v1)  


## TransformerST
[TransformerST](https://doi.org/10.1101/2022.08.11.503261)  


## FICT
[FICT](https://doi.org/10.1093/bioinformatics/btab704)    



# 目前整理到可以使用的数据集

## 