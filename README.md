# StyleSDF

## [Paper](https://repository.tudelft.nl/islandora/object/uuid:7f8ef49b-7c9c-4281-bd93-b921d9b28d49/datastream/OBJ/download) | [Data]()

![Alt Text](./media/bkgogh.gif)

## Description

This repository contains the project created for my Masters thesis [Neural 3D Reconstruction and Stylization](https://repository.tudelft.nl/islandora/object/uuid:7f8ef49b-7c9c-4281-bd93-b921d9b28d49?collection=education).

Using [IDR](https://lioryariv.github.io/idr/), we create a 3D reconstruction which we then style using NNFM from [ARF](https://www.cs.cornell.edu/projects/arf/) or the original [NST](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) style approach.

## Installation Requirements

The code was run using python 3.9, and requires packages which can be installed by running
```
pip install -r requirements.txt
```

## Usage

Create a 3D reconstruction using IDR of the BK dataset by running

```
cd opt
python run_IDR.py --conf DTU_style.conf --scan_id 105
```

Once this has been completed the result is stored in the created exp folder.
The result can be styled by running

```
cd opt
python run_style.py --conf DTU_style.conf --scan_id 24 --style vangogh_starry_night
```
## Datasets


## Preprocessing

WIP
