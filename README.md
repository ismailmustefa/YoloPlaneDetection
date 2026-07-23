# RarePlanes Aircraft Detection with YOLO

This project trains a YOLOv8 object-detection model to locate and classify aircraft in satellite imagery from the [RarePlanes dataset](https://www.iqt.org/library/the-rareplanes-dataset). The complete workflow is contained in [`371Final.ipynb`](./371Final.ipynb) and is designed to run in Google Colab with the dataset stored in the user's own Google Drive.

## What the project does

RarePlanes provides tiled satellite images and aircraft annotations in GeoJSON format. YOLO cannot train directly from those geographic polygon annotations, so this project builds the required training pipeline:

1. Mounts Google Drive and locates the two RarePlanes training archives.
2. Safely extracts the image and annotation archives into temporary runtime storage.
3. Matches every GeoJSON annotation file to its corresponding PNG image tile.
4. Converts geographic aircraft polygons into normalized YOLO bounding-box labels.
5. Visually compares the converted boxes with the original GeoJSON annotations to check the conversion.
6. Splits the matched images and labels into reproducible training and validation sets.
7. Fine-tunes a pretrained YOLOv8 medium model on the prepared dataset.
8. Validates the trained model and tests predictions at several confidence thresholds.
9. Copies the best model checkpoint to Google Drive so it remains available after the Colab session ends.

## Aircraft classes

The annotation conversion uses the RarePlanes `role` and `propulsion` properties to create seven model classes:

| ID | Class |
|---:|---|
| 0 | `civilian_jet` |
| 1 | `civilian_propeller` |
| 2 | `civilian_unpowered` |
| 3 | `military_jet` |
| 4 | `military_propeller` |
| 5 | `military_unpowered` |
| 6 | `unknown` |

The `unknown` class preserves annotations whose role or propulsion values do not match the expected categories.

## How the workflow was accomplished

The notebook reads each image's geospatial bounds with Rasterio and uses those bounds to translate annotation coordinates into image-pixel coordinates. Each polygon is reduced to a rectangular bounding box and then normalized to YOLO's five-value label format:

```text
class_id x_center y_center width height
```

The converted labels are checked in two ways. First, the notebook prints a generated label file so its structure can be inspected. It then draws the YOLO bounding boxes and the original GeoJSON-derived boxes beside the source image. Additional randomly selected samples help expose coordinate scaling, orientation, or alignment problems before training begins.

For training, the notebook pairs each image with its converted label and creates the directory layout expected by Ultralytics. A fixed random seed produces a repeatable training/validation split. It then fine-tunes `yolov8m.pt` for up to 50 epochs at 640 × 640 resolution using AdamW and early stopping. After training, the notebook reports validation metrics and compares detections at confidence thresholds of 0.10, 0.25, 0.50, and 0.75.

## Run the project with your own Google Drive

### 1. Download the dataset

Download the RarePlanes training data from the [official dataset page](https://www.iqt.org/library/the-rareplanes-dataset). The notebook expects these two files with their original names:

```text
RarePlanes_train_PS-RGB_tiled.tar.gz
RarePlanes_train_geojson_aircraft_tiled.tar.gz
```

### 2. Create a folder in Google Drive

Create a folder in `My Drive` for the project and upload both archives to it. For example:

```text
My Drive/
└── YoloPlaneDetection/
    └── rareplanes/
        ├── RarePlanes_train_PS-RGB_tiled.tar.gz
        └── RarePlanes_train_geojson_aircraft_tiled.tar.gz
```

Uploading and extracting these large archives can take time. Keep the archives compressed in Drive—the notebook extracts them into the Colab runtime automatically.

### 3. Open the notebook in Google Colab

Download or clone this repository, then open `371Final.ipynb` in [Google Colab](https://colab.research.google.com/). A GPU is strongly recommended for model training. In Colab, choose **Runtime → Change runtime type → GPU** before running the training section.

### 4. Set the dataset path

After mounting Drive, find the notebook cell that defines `data_path` and replace the existing project-specific path with the folder you created. Using the example above:

```python
data_path = "/content/drive/MyDrive/YoloPlaneDetection/rareplanes"
```

Run the path and archive checks immediately below it. Both checks should confirm that the files exist before you continue.

### 5. Set the model output folder

Near the end of the training section, find the cell that defines `save_folder`. Change it to a folder in your own Drive where the trained checkpoint should be kept:

```python
save_folder = "/content/drive/MyDrive/YoloPlaneDetection"
```

The notebook will copy the best checkpoint there as:

```text
best_rareplanes_type_propulsion_model.pt
```

### 6. Run the notebook in order

Use **Runtime → Run all**, or run each section from top to bottom. The required order is:

1. Imports and Drive mounting
2. Dataset checks and extraction
3. Annotation inspection and label conversion
4. Visual label verification
5. Dataset preparation
6. Ultralytics installation and model training
7. Validation, checkpoint saving, and prediction testing

If an import is unavailable in the selected Colab runtime, install the missing package in a cell with `pip` and rerun the import cell. Ultralytics is installed by the notebook before training.

## Storage and generated files

Images, converted labels, dataset splits, and most prediction outputs are created in the Colab runtime's temporary storage. They disappear when the runtime is reset or disconnected. The notebook explicitly copies only the best trained model to the configured Google Drive folder.

To preserve any other output—such as generated labels, plots, or prediction images—copy it to Google Drive before ending the session.

## Repository contents

- [`371Final.ipynb`](./371Final.ipynb) — data conversion, verification, training, validation, and testing workflow
- [`README.md`](./README.md) — project explanation and setup instructions
