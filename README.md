### Classification of time series data using techniques on the Grassmann manifold

[Poster](https://github.com/adamguos/manifold/blob/master/poster/poster_final.pdf)

Uses the autoregressive-moving average (ARMA) model to parametrise time series data, then represents
the ARMA parameters as points on a Grassmann manifold (collection of Euclidean subspaces). Performs
classification using a support vector machine (SVM) equipped with kernel methods defined on
Grassmann manifolds.

Performs well on high-dimensional raw data.

#### Results

Details on datasets in poster:

| Dataset       | Grassmann | Literature                                            |
-------------------------------------------------------------------------------------
| Alcohol EEG   | 99.8%     | [97.1%](https://doi.org/10.1007/s10489-017-1042-9)    |
| Vehicle audio | 62.8%     | [88.2%](http://arxiv.org/abs/1705.09869)              |
| Video digits  | 97.0%     | [94.7%](https://doi.org/10.1007/978-0-8176-8095-4_11) |

#### Citations

- [Turaga et al.](https://doi.org/10.1109/TPAMI.2011.52): Statistical Computations on Grassmann and
  Stiefel Manifolds for Image and Video-Based Recognition
- [Jayasumana et al.](https://doi.org/10.1109/TPAMI.2015.2414422): Kernel Methods on Riemannian
  Manifolds with Gaussian RBF Kernels
