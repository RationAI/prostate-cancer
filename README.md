# Prostate Cancer

**[RationAI, FAIR](https://rationai.fi.muni.cz)**

Michal Jakubík, Matěj Gallo, Vít Musil, Matěj Pekár

An implementation of a pipeline for training a binary
classifier to detect prostate cancer using Whole Slide Images (WSIs). Built with
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), [Hydra](https://hydra.cc)
and [MlFlow](https://mlflow.org).

Due to the unavailability of the dataset used in this project, users will
need to create their own data source to utilize the pipeline.


## Installation

To install the necessary dependencies, follow these steps:

1. Install [PDM](https://github.com/pdm-project/pdm)
2. Clone the repository.
3. Install the dependencies using PDM.


```bash
git clone git@github.com:rationai/prostate-caner.git
cd prostate-cancer
pdm install
```

## Getting Started

### Implementing Custom Tile Extraction

First, implement your custom extract_tile method in `prostate_cancer/datamodule/datasets/base_wsi.py`:

```python
def extract_tile(
    slide_fp: Path, coord_x: int, coord_y: int, tile_size: int, level: int
) -> NDArray[np.uint8]:
    """Extracts a tile from a slide using the supplied coordinate values.

    Args:
        slide_fp (Path): Path to the slide.
        coord_x (int): Coordinates of a tile to be extracted at OpenSlide level 0 resolution.
        coord_y (int): Coordinates of a tile to be extracted at OpenSlide level 0 resolution.
        tile_size (int): Size of the tile to be extracted.
        level (int): Resolution level from which tile should be extracted.

    Returns:
        NDArray: RGB Tile represented as numpy array.
    """
```

### Creating Custom Data Source
Next, create a custom data source by subclassing the `BaseDataSource` class in `prostate_cancer/datamodule/datasources.py`.

### Updating Configuration Files
Finally, add your custom data sources to the configuration files (`train.yaml`, `train_without_valid.yaml`, `test.yaml`) located in the `conf/experiment/cancer_bc` directory.


## Training the Model

To train the model, run the following command:
```bash
pdm train +experiment=cancer_bc/train
```


## License

The project is licensed under the [MIT license](LICENSE).
