# quadtree_kde

## Project Description
This project combines the gaussian_kde implementation from [IceCubeOpenSource](https://github.com/IceCubeOpenSource/kde) with the adaptive quadtree concept from [FaLi-KunxiaojiaYuan](https://github.com/FaLi-KunxiaojiaYuan/Spatial-Statistics)

By using the quadtree to assign a bandwidth to each of the data points an approximate 60x speedup is achieved on our systems as compared to the *cuda* adaptive implementation.  In other words, for a data set that required 10 hours to compute the adaptive bandwidth factors using ```gaussian_kde```, ```quadtree_kde``` takes 10 minutes.

## Use
1. Install pykde in your system
```bash
pip install git+https://github.com/icecubeopensource/kde.git#egg=kde[cuda]
```
and set up the cuda dependencies as necessary.
2. Clone this repository and the submodule
