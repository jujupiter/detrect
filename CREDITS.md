# Credits

The code in this repository is under the MIT licence (see `LICENSE`). The images are not in the repository:
the scripts download them from their sources, and each source keeps its own licence. Anything that uses the
data or a model trained on it (a published model, a demo, an app) should carry the credits below.

## Training and test data

| Dataset | Used for | Licence | Credit |
|---|---|---|---|
| [trashnet](https://github.com/garythung/trashnet) | training, test | MIT | Gary Thung and Mindy Yang, 2016 |
| [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) | training, test | CC BY 4.0 | Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G |
| [TACO](http://tacodataset.org) | training, test | annotations CC BY 4.0; images under their own free licence (see below) | Proença, P. F., & Simões, P. (2020). TACO: Trash Annotations in Context for Litter Detection. arXiv:2003.06975 |
| Wikimedia Commons photos | test only | per photo (CC0, public domain, CC BY, CC BY-SA) | per photo: title, author and licence in `data/commons_*/sources.csv` |

TACO images: the licence of each image is in TACO's annotation file. Images with no licence entry are CC BY 4.0
by default (715 of 1,500); 466 come from OpenLitterMap under the ODbL ("© OpenLitterMap & Contributors") and 319
are marked "CC".

## Reference data

| Data | Licence | Credit |
|---|---|---|
| EU Joint List of Litter Categories (`data/jlist.csv`, J-codes) | European Commission reuse policy: CC BY 4.0 for content owned by the EU | Fleet, D., Vlachogianni, Th., & Hanke, G. (2021). A Joint List of Litter Categories for Marine Macrolitter Monitoring. EUR 30348 EN, Publications Office of the European Union. https://doi.org/10.2760/127473 |

## Model weights

The models start from ImageNet weights provided by [torchvision](https://github.com/pytorch/vision) (BSD 3-Clause).

## Datasets mapped to J-codes but not used yet

`data/extra_jcodes.csv` maps their classes to J-codes. Credit them as below if they are added to training.

| Dataset | Licence | Credit |
|---|---|---|
| [Drinking Waste Classification](https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification) | CC0 1.0 | Arkadiy Serezhkin (no credit required, but welcome) |
| [FloatWaste](https://huggingface.co/datasets/OliviaWuYJ/sc-floatwaste) | CC BY 4.0 | Wu, Y. et al. FloatWaste: a multi-region riverine floating-waste detection benchmark (full citation pending) |
| [pLitterStreet](https://zenodo.org/records/8288500) | CC BY 4.0 | pLitterStreet - Street Level Plastic Litter Detection Dataset, https://doi.org/10.5281/zenodo.8288500 |
| [AquaPlasticWaste 2026](https://data.mendeley.com/datasets/zshjxgfmc3/1) | CC BY 4.0 | Veningston K., Ahsan Ul Haq, Shafiya Mushtaq, Dinesh Kumar R. (2026). https://doi.org/10.17632/zshjxgfmc3.1 |
