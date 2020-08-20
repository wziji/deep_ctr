today=`date '+%Y-%m-%d %H:%M:%S'`
echo $today



sh split_train_val.sh

python write_tfrecord.py

sh run_train_esmm_model.sh
