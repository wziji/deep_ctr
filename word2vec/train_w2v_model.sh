# trainData: skus, split by space

trainData=data/w2v_order_seq
outputData=data/w2v_order_item_vec_128dim



size=128
window=5
iter=20
classes=0
binary=0
threads=22
min_count=8
negative=25


if [ -s $trainData ]; then

    # train model
    ./word2vec -train $trainData -output $outputData -size $size -window $window -sample 1e-4 -negative $negative -hs 0 -binary $binary -cbow 0 -iter $iter -threads $threads -classes $classes -min_count $min_count 

    # Delete the top 2 lines
    sed -i 1,2d $outputData
    sed -i "s/ /\t/g" $outputData
fi


# cbow (-cbow 1)
# skip (-cbow 0)