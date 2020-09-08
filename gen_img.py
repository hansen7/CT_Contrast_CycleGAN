import os, pdb, json, numpy as np, pandas as pd
import nibabel as nib, matplotlib.pyplot as plt
from skimage import transform
from tqdm import tqdm


if __name__ == "__main__":

    all_meta =  pd.read_csv('COVID-19/cam_clinical_all.csv')

    def load_image(image_path):
        image = nib.load(image_path).get_fdata()

        h_train, w_train, z_train = image.shape
        if h_train != 512 or w_train != 512:
            image = transform.resize(image, (512, 512))
        return z_train, image


    contrast_cts = all_meta.loc[all_meta['Contrast'] == 1.0]['SlideID'].values
    np.random.shuffle(contrast_cts)
    train_contrast_cts = contrast_cts[:len(contrast_cts) * 4//5]
    test_contrast_cts = contrast_cts[len(contrast_cts) * 4//5:]

    nocontrast_cts = all_meta.loc[all_meta['Contrast'] != 1.0]['SlideID'].values
    np.random.shuffle(nocontrast_cts)
    train_nocontrast_cts = nocontrast_cts[:len(nocontrast_cts) * 4//5]
    test_nocontrast_cts = nocontrast_cts[len(nocontrast_cts) * 4//5:]

    print("Contrast Train")

    for slideid in tqdm(train_contrast_cts):
        z_train, img = load_image(os.path.join('COVID-19', 'NHSX', 'nifti', slideid))
        if z_train < 60:
            continue
        selected_idx = np.random.randint(low=10, high=z_train, size=2)
        for idx in selected_idx:
            plt.imsave(os.path.join("pytorch-CycleGAN-and-pix2pix-master", 
                                    "datasets", 
                                    "contrast2no", 
                                    "trainA",
                                    slideid.replace(".nii.gz", "_" + str(idx)) + ".png"), 
                       img[:,:,idx])
     
     
    print("Contrast Test")
    for slideid in tqdm(test_contrast_cts):
        z_train, img = load_image(os.path.join('COVID-19', 'NHSX', 'nifti', slideid))
        if z_train < 60:
            continue
        selected_idx = np.random.randint(low=10, high=z_train, size=2)
        for idx in selected_idx:
           plt.imsave(os.path.join("pytorch-CycleGAN-and-pix2pix-master", 
                                   "datasets", 
                                   "contrast2no", 
                                   "testA", 
                                   slideid.replace(".nii.gz", "_" + str(idx)) + ".png"), 
                      img[:,:,idx])


    print("noConrast Train")
    for slideid in tqdm(train_nocontrast_cts):
       z_train, img = load_image(os.path.join('COVID-19', 'NHSX', 'nifti', slideid))
       if z_train < 60:
           continue
       selected_idx = np.random.randint(low=10, high=z_train, size=5)
       for idx in selected_idx:
           plt.imsave(os.path.join("pytorch-CycleGAN-and-pix2pix-master",
                                   "datasets", 
                                   "contrast2no", 
                                   "trainB", 
                                   slideid.replace(".nii.gz", "_" + str(idx)) + ".png"), 
                      img[:,:,idx])

    print("noConrast Test")
    for slideid in tqdm(test_nocontrast_cts):
        z_train, img = load_image(os.path.join('COVID-19', 'NHSX', 'nifti', slideid))
        if z_train < 60:
            continue
        selected_idx = np.random.randint(low=10, high=z_train, size=5)
        for idx in selected_idx:
            plt.imsave(os.path.join("pytorch-CycleGAN-and-pix2pix-master", 
                                    "datasets", 
                                    "contrast2no", 
                                    "testB", 
                                    slideid.replace(".nii.gz", "_" + str(idx)) + ".png"), 
                                    img[:,:,idx])

