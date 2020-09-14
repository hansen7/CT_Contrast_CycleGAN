
import os, numpy as np, argparse, nibabel as nib, matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_id', type=str, help='nifit id [default: ]')
    parser = parser.parse_args()
    
    ct_id = parser.ct_id  # ct_id = 'Covid4238_20170112_0_0'
    nifti_file = os.path.join('./COVID-19/NHSX/nifti', ct_id + '.nii.gz')
    ctimg = nib.load(nifti_file)

    output_dir = os.path.join('datasets/contrast2no_new', ct_id)
    os.makedirs(output_dir, exist_ok=True)

    ct_arrays = np.array(ctimg.get_fdata())
    _, _, z_num = ct_arrays.shape
    for idx in range(z_num):
        plt.imsave(os.path.join(output_dir, str(idx) + ".jpg"), ct_arrays[:,:,idx])

    result_dir = os.path.join('results', ct_id)

    exe = 'python test.py ' + '--gpu_ids 6 ' + '--name contrast2no_new ' \
        + '--dataroot %s '%output_dir + '--results_dir %s '%result_dir \
        + '--input_nc 1 --output_nc 1 --load_size 512 --crop_size 512 --model test --no_dropout'

    # os.system(exe)

