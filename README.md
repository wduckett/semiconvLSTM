## Installation

Create a conda environment first.

```bash
conda create -n semiconv python=3.9.6
conda activate semiconv
```

### Install semiconv

```bash
pip install -r requirements.txt     # developer tools
python setup.py install
```
### Preprocess the data
```
python semiconv/preprocess.py --meta_point_path Data/meta_point.csv --scoot_detector_path Data/scoot_detector.csv --scoot_data_path Data/SCOOT.csv --output_path Data/data.npy --slice_size 168 --overlap 167
```
### Run training

```
python semiconv/train.py --data_path path/to/processedData.npy --log_file result.csv --checkpoint_path checkpoint.model.keras --model_path model.tf --learning_rate 0.001 --epochs 100 --batch_size 64
```

### Run inference

```
python semiconv/forecast.py --data_path Data/data.npy --sensor_mask_path Data/SensorMask.npy --model_path pretrainedModels/semiconv32to32.tf --output_path Forecasts/forecast.npy
```