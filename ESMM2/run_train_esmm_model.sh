train_data_path=data/train_data.tfrecord
val_data_path=data/val_data.tfrecord
model_path=__model/esmm_finetune.model
train_val_summary_path=data/train_val_summary

python train_esmm_finetune.py $train_data_path $val_data_path $model_path $train_val_summary_path


# tar model
model_dir=__model
python tar_model.py $model_dir
