
echo "Start!"

cd ..
while read ids; 
do
    echo "$ids";
    python convert_img.py --ct_id $ids ;
    python test.py --gpu_ids 0 \
        --dataroot datasets/contrast2no_new/$ids \
        --results_dir results/$ids \
        --name contrast2no_new \
        --load_size 512 \
        --crop_size 512 \
        --output_nc 1 \
        --input_nc 1 \
        --model test \
        --no_dropout ;
    python gen_nifti.py --ct_id $ids ;
done < filelists/subset.txt
