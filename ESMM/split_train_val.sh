raw_sample_data_path=data/esmm_raw_sample_data
sample_data_path=data/esmm_sample_data

# shuffle
echo `date`
echo 'Start to shuffle sample data ...'
shuf $raw_sample_data_path > $sample_data_path
echo `date`
echo 'Shuffle Done!'



# split train and val
train_path=data/train_data
val_path=data/val_data
summary_path=data/train_val_summary
train_percent=0.75 # percent in (0, 1)

echo 
echo `date`
echo 'Start to split train and validate data ...'
python split_train_val.py $sample_data_path $train_path $val_path $summary_path $train_percent
echo `date`
echo 'Split train and validate data Done!'
