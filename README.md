# StyleSDF

## [Paper](https://repository.tudelft.nl/islandora/object/uuid:7f8ef49b-7c9c-4281-bd93-b921d9b28d49/datastream/OBJ/download) | [Project Page](https://fabisser.github.io/pages/styleSDF/index.html)

![Alt Text](./media/gogh.gif)

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

[Dataset](https://www.dropbox.com/sh/5tam07ai8ch90pf/AADniBT3dmAexvm_J1oL__uoa) from IDR.

We have also created a dataset of the TU Delft Architecture building, included in the repository.

## Preprocessing

To create your own dataset, camera information is required which can be calculated using COLMAP. As input the following is required

```
- <data_folder>
    - images      # input images
    - masks       # mask data of the input images
```

Run COLMAP using the following command-lines


```
colmap feature_extractor \
    --database_path <data_folder>/database.db \
    --image_path <data_folder>/image

colmap exhaustive_matcher \
    --database_path <data_folder>/database.db

mkdir <data_folder>/sparse

colmap mapper \
    --database_path <data_folder>/database.db \
    --image_path <data_folder>/image \
    --output_path <data_folder>/sparse

colmap model_converter \
    --input_path <data_folder>/sparse/0 \
    --output_path <data_folder>/sparse \
    --output_type TXT
```
To create the required camera paremeters as well as normalized camera parameters (see IDR), run the following
```
python colmap2idr.py --dense_folder <data_folder> --max_d 256 --convert_format

python3 preprocess_cameras.py --source_dir <data_folder>
```
