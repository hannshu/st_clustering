[中文](README.md) | English

# CONTENT
- [CONTENT](#content)
- [Clustering method for single cell spatial transcriptomics data](#clustering-method-for-single-cell-spatial-transcriptomics-data)
- [Performance comparison](#performance-comparison)
  - [Datasets](#datasets)
  - [Performance](#performance)
    - [Testing device](#testing-device)
- [NOTICE](#notice)

# Clustering method for single cell spatial transcriptomics data
| model name | cluster method | published time | method/layer                                        | published                                           | source                                              | Github Repo                                       |
|----------------|--------------------|--------------------|---------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------|
| Louvain        | Louvain            | 2008-09-10         | Community Discovery                                     | Journal of Statistical Mechanics: Theory and Experiment | [DOI](https://doi.org/10.1088/1742-5468/2008/10/P10008) | -                                                     |
| stLearn        |                    | 2020-05-31         |                                                         | bioRxiv                                                 | [DOI](https://doi.org/10.1101/2020.05.31.125658)        |                                                       |
| Seurat         |                    | 2021-06-24         |                                                         | Cell                                                    | [DOI](https://doi.org/10.1016/j.cell.2021.04.048)       |                                                       |
| SEDR           |                    | 2021-07-06         |                                                         | Research Square                                         | [DOI](https://doi.org/10.21203/rs.3.rs-665505/v1)       |                                                       |
| BayesSpace     |                    | 2021-11            | static                                                  | Nature Biotechnology                                    | [DOI](https://doi.org/10.1038/s41587-021-00935-2)       |                                                       |
| SpaGCN         | self-build (DEC)   | 2021-11            | GNN/GCN                                                 | Nature Methods                                          | [DOI](https://doi.org/10.1038/s41592-021-01255-8)       | [repo](https://github.com/jianhuupenn/SpaGCN.git)     |
| Giotto         |                    | 2021-12            |                                                         | BMC Genome Biology                                      | [DOI](https://doi.org/10.1186/s13059-021-02286-2)       |                                                       |
| FICT           |                    | 2022-01-27         | static/EM                                               | Bioinformatics                                          | [DOI](https://doi.org/10.1093/bioinformatics/btab704)   |                                                       |
| STAGATE        | mclust-EEE*        | 2022-04-01         | GNN/AutoEncoder+GAT                                     | Nature Communications                                   | [DOI](https://doi.org/10.1038/s41467-022-29439-6)       | [repo](https://github.com/QIFEIDKN/STAGATE_pyG.git)   |
| CCST           | UMAP/Kmeans        | 2022-06-07         | GNN/DGI                                                 | Nature Computational Science                            | [DOI](https://doi.org/10.1038/s43588-022-00266-5)       | [repo](https://github.com/xiaoyeye/CCST.git)          |
| DR-SC          |                    | 2022-07-08         |                                                         | Nucleic Acids Research                                  | [DOI](https://doi.org/10.1093/nar/gkac219)              |                                                       |
| DeepST         | Leiden             | 2022-12-09         | GNN/AutoEncoder+GCN                                     | Nucleic Acids Research                                  | [DOI](https://doi.org/10.1093/nar/gkac901)              | [repo](https://github.com/JiangBioLab/DeepST.git)     |
| GraphST        | mclust-EEE*        | 2023-03-01         | GNN/GCN  (AutoEncoder_Loss + contrastive_learning_loss) | Nature Communication                                    | [DOI](https://doi.org/10.1038/s41467-023-36796-3)       | [repo](https://github.com/JinmiaoChenLab/GraphST.git) |
*mclust version 5.4.10 is used in this testing.

# Performance comparison    
## Datasets
| dataset name            | tech            | source                                                                                                    | size(spots*genes) |
|-----------------------------|---------------------|---------------------------------------------------------------------------------------------------------------|-----------------------|
| human breast cancer dataset | 10x Genomics Visium | [link](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1)     | 3798*36601            |
| DLPFC                       | 10x Genomics Visium | [link](https://github.com/LieberInstitute/spatialLIBD)                                                                | (3460-4789)*33538   |
| 10x visium hne              | 10x Genomics Visium | [link](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Adult_Mouse_Brain)                   | 2688*18078            |
| 10x visium fluo             | 10x Genomics Visium | [link](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Adult_Mouse_Brain_Coronal_Section_2) | 2800*16562            |

- [ST datasets](https://github.com/hannshu/st_datasets.git) 

## Performance
To help you with further testing, all trained embedding matrices or predict labels are saved in [百度网盘](https://pan.baidu.com/s/1SUYZIO2vsU08u7yqTT8Oww?pwd=tkmn), [Google Drive](https://drive.google.com/drive/folders/1FIFdXh1oNQSpgq20G_QHuFLRjEL_OHCF?usp=sharing).
[Performance Comparison Conclusion](result.csv)

### Testing device
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
- If the model provides a pypi install script, we will install the package and test it.
- If the model does not provide a setup script, we may modify the model's script to fit our testing input. All modified scripts will be provided in this repository.
- While using the same `SEED`, it may appear differently on different GPU devices. So if you find a different result on your device, it is an ordinary phenomenon.
- Models using Pytorch Geometric (PyG) may produce different results even if you are using the same device. To see detailed information, see [issue #2788](https://github.com/pyg-team/pytorch_geometric/issues/2788#issuecomment-870502307)  
- If you find anything wrong, you are very welcome to submit an issue or contact me [via email](mailto:hanshu.npu@gmail.com).
