中文 | [English](README_en.md)

# 目录
- [目录](#目录)
- [整理到的对空间转录组进行聚类分析的算法](#整理到的对空间转录组进行聚类分析的算法)
- [性能比较](#性能比较)
  - [数据集](#数据集)
  - [效果对比](#效果对比)
    - [测试设备](#测试设备)
- [NOTICE](#notice)

# 整理到的对空间转录组进行聚类分析的算法
| model name | cluster method | published time | method/layer                                        | published                                           | source                                              | intro                                                                                                                        | Github Repo                                       |
|----------------|--------------------|--------------------|---------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| Louvain        | Louvain            | 2008-09-10         | Community Discovery                                     | Journal of Statistical Mechanics: Theory and Experiment | [DOI](https://doi.org/10.1088/1742-5468/2008/10/P10008) | [intro](https://blog.hanshu.org/%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84/louvain,-STAGATE,-CCST,-SpaGCN%E6%80%BB%E7%BB%93/) | -                                                     |
| stLearn        |                    | 2020-05-31         |                                                         | bioRxiv                                                 | [DOI](https://doi.org/10.1101/2020.05.31.125658)        | NULL                                                                                                                             |                                                       |
| Seurat         |                    | 2021-06-24         |                                                         | Cell                                                    | [DOI](https://doi.org/10.1016/j.cell.2021.04.048)       | NULL                                                                                                                             |                                                       |
| SEDR           |                    | 2021-07-06         |                                                         | Research Square                                         | [DOI](https://doi.org/10.21203/rs.3.rs-665505/v1)       | NULL                                                                                                                             |                                                       |
| BayesSpace     |                    | 2021-11            | static                                                  | Nature Biotechnology                                    | [DOI](https://doi.org/10.1038/s41587-021-00935-2)       | NULL                                                                                                                             |                                                       |
| SpaGCN         | self-build (DEC)   | 2021-11            | GNN/GCN                                                 | Nature Methods                                          | [DOI](https://doi.org/10.1038/s41592-021-01255-8)       | [intro](https://blog.hanshu.org/%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84/louvain,-STAGATE,-CCST,-SpaGCN%E6%80%BB%E7%BB%93/) | [repo](https://github.com/jianhuupenn/SpaGCN.git)     |
| Giotto         |                    | 2021-12            |                                                         | BMC Genome Biology                                      | [DOI](https://doi.org/10.1186/s13059-021-02286-2)       | NULL                                                                                                                             |                                                       |
| FICT           |                    | 2022-01-27         | static/EM                                               | Bioinformatics                                          | [DOI](https://doi.org/10.1093/bioinformatics/btab704)   | NULL                                                                                                                             |                                                       |
| STAGATE        | mclust-EEE*        | 2022-04-01         | GNN/AutoEncoder+GAT                                     | Nature Communications                                   | [DOI](https://doi.org/10.1038/s41467-022-29439-6)       | [intro](https://blog.hanshu.org/%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84/louvain,-STAGATE,-CCST,-SpaGCN%E6%80%BB%E7%BB%93/) | [repo](https://github.com/QIFEIDKN/STAGATE_pyG.git)   |
| CCST           | UMAP/Kmeans        | 2022-06-07         | GNN/DGI                                                 | Nature Computational Science                            | [DOI](https://doi.org/10.1038/s43588-022-00266-5)       | [intro](https://blog.hanshu.org/%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84/louvain,-STAGATE,-CCST,-SpaGCN%E6%80%BB%E7%BB%93/) | [repo](https://github.com/xiaoyeye/CCST.git)          |
| DR-SC          |                    | 2022-07-08         |                                                         | Nucleic Acids Research                                  | [DOI](https://doi.org/10.1093/nar/gkac219)              | NULL                                                                                                                             |                                                       |
| DeepST         | Leiden             | 2022-12-09         | GNN/AutoEncoder+GCN                                     | Nucleic Acids Research                                  | [DOI](https://doi.org/10.1093/nar/gkac901)              | [intro](https://blog.hanshu.org/%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84/DeepST/)                                           | [repo](https://github.com/JiangBioLab/DeepST.git)     |
| GraphST        | mclust-EEE*        | 2023-03-01         | GNN/GCN  (AutoEncoder_Loss + contrastive_learning_loss) | Nature Communication                                    | [DOI](https://doi.org/10.1038/s41467-023-36796-3)       | [intro](https://blog.hanshu.org/%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84/GraphST/)                                          | [repo](https://github.com/JinmiaoChenLab/GraphST.git) |
*测试时使用的mlcust版本为5.4.10  

# 性能比较  
## 数据集
| dataset name            | tech            | source                                                                                                    | size(spots*genes) |
|-----------------------------|---------------------|---------------------------------------------------------------------------------------------------------------|-----------------------|
| human breast cancer dataset | 10x Genomics Visium | [link](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1)     | 3798*36601            |
| DLPFC                       | 10x Genomics Visium | [link](https://github.com/LieberInstitute/spatialLIBD)                                                                | (3460-4789)*33538   |
| 10x visium hne              | 10x Genomics Visium | [link](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Adult_Mouse_Brain)                   | 2688*18078            |
| 10x visium fluo             | 10x Genomics Visium | [link](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Adult_Mouse_Brain_Coronal_Section_2) | 2800*16562            |

- [空间转录组数据集](https://github.com/hannshu/st_datasets.git) 

## 效果对比
为方便您进行后续的测试，模型训练得到的embedding或预测标签保存在[百度网盘](https://pan.baidu.com/s/1SUYZIO2vsU08u7yqTT8Oww?pwd=tkmn), [Google Drive](https://drive.google.com/drive/folders/1FIFdXh1oNQSpgq20G_QHuFLRjEL_OHCF?usp=sharing)中  
[模型效果对比总结](result.csv)

### 测试设备
1. Intel Core i5-12600KF
2. Intel Core i7-13700K, Nvidia RTX 2080Ti
3. Intel Xeon(R) Gold 6226R, Nvidia TITAN RTX

| model name    | device |
|---------------|--------|
| K-means       | 3      |
| Louvain       | 3      |
| stLearn       | 3      |
| Seurat        |        |
| SEDR          | 3      |
| BayesSpace    | 1      |
| SpaGCN        | 3      |
| Giotto        |        |
| FICT          |        |
| STAGATE       | 3      |
| CCST          | 3      |
| DR-SC         |        |
| TransformerST |        |
| DeepST        | 2      |
| GraphST       | 3      |

# NOTICE
- 如果模型提供了pypi包安装脚本，则会安装为pypi包后进行测试。
- 如果模型没有提供setup脚本，则在测试时可能对模型脚本进行连部分修改，所有经过修改脚本的模型的源码都会放在本仓库中。  
- 在不同gpu上同一SEED依然可能产生不同的随机数，在您的设备上可能得到不同的测试结果。  
- 上述部分使用Pytorch Geometric(PyG)框架的方法由于pytorch_scatter包在gpu测试时会产生不受SEED控制的随机数，所以使用到pytorch_scatter包的模型在不同轮次的训练中产生的embedding可能不相同，导致测试结果可能不相同。详见pyg作者对[issue的回答](https://github.com/pyg-team/pytorch_geometric/issues/2788#issuecomment-870502307)
- 如有错误，恳请您提出issue或[邮件与我联系](mailto:hanshu.npu@gmail.com)。
- 部分方法的中文简介在[我的blog](https://blog.hanshu.org/%E7%A9%BA%E9%97%B4%E8%BD%AC%E5%BD%95%E7%BB%84/%E8%81%9A%E7%B1%BB%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87/)中。
