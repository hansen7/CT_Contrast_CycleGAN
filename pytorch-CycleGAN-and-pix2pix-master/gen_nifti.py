
import os, numpy as np, argparse, shutil, nibabel as nib, matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ct_id', type=str, help='nifit id [default: ]')
    parser = parser.parse_args()
    ct_id = parser.ct_id  # ct_id = 'Covid4238_20170112_0_0'
    
    nifti_filename = os.path.join('./nhsx/nifti', ct_id + '.nii.gz')
    ctimg = nib.load(nifti_filename)
    ct_arrays = ctimg.get_fdata()
    _, _, z_num = ct_arrays.shape
    new_data = ct_arrays.copy()

    img_dir = os.path.join('results', ct_id, 'contrast2no_new', 'test_latest', 'images')
    
    for idx in range(z_num):
        new_data[:,:,idx] = plt.imread(os.path.join(img_dir, '%d_fake.png'%idx))[:,:,0]

    ct_out_dir = './nhsx/nifti_CycleGAN_Converted'
    os.makedirs(ct_out_dir, exist_ok=True)

    # ref: https://bic-berkeley.github.io/psych-214-fall-2016/saving_images.html
    clipped_img = nib.Nifti1Image(new_data, ctimg.affine, ctimg.header)
    nib.save(clipped_img, os.path.join(ct_out_dir, ct_id + '.nii.gz'))
    shutil.rmtree('datasets/contrast2no_new/%s' % ct_id)
    shutil.rmtree('results/%s' % ct_id)
    print(ct_id + '.nii.gz generated')
    print('='*33, '\n\n\n')
