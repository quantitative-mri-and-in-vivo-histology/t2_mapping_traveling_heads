import os
from bids.layout import BIDSLayout
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from nipype.interfaces import fsl
import scipy


# Define the slice positions for the central axial, sagittal, and coronal slices
def get_central_slices(volume):
    x_mid = (volume.shape[0] // 2) - 5  # Sagittal (cut along x-axis)
    y_mid = volume.shape[1] // 2  # Coronal (cut along y-axis)
    z_mid = volume.shape[2] // 2  # Axial (cut along z-axis)

    # Extract the 2D slices
    sagittal_slice = volume[x_mid, :, :]
    coronal_slice = volume[:, y_mid, :]
    axial_slice = volume[:, :, z_mid]

    return sagittal_slice, coronal_slice, axial_slice


if __name__ == "__main__":

    out_dir_base = "../../data/figures/qualitative"
    same_scaling = False
    if same_scaling:
        out_dir = f"{out_dir_base}/same_scaling"
    else:
        out_dir = f"{out_dir_base}/scaled_per_scan_alt"

    os.makedirs(out_dir, exist_ok=True)

    brain_mask_file = fsl.Info.standard_image('MNI152_T1_1mm_brain_mask.nii.gz')
    brain_mask = nib.load(brain_mask_file).get_fdata()

    brain_probseg_file = "/home/laurin/workspace/t2_mapping_traveling_heads/data/atlases/space-MNI152_label-brain_desc-SPM_probseg.nii.gz"
    brain_probseg = nib.load(brain_probseg_file).get_fdata()
    brain_mask = brain_probseg > 0.95

    brain_mask = nib.load("/usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz").get_fdata()

    dataset_dirs = [
        "/media/laurin/Data_share/Travel_Head_Study/ismrm_dataset_2025/results/Bonn_Skyra_3T_LowRes_bids",
        "/media/laurin/Data_share/Travel_Head_Study/ismrm_dataset_2025/results/Hamburg_Prisma_3T_bids_3depi",
        "/media/laurin/Data_share/Travel_Head_Study/ismrm_dataset_2025/results/London_Kings_Vida_3T_bids",
        "/media/laurin/Data_share/Travel_Head_Study/ismrm_dataset_2025/results/Hamburg_Prisma_3T_bids_ssfp"
    ]
    dataset_names = [
        "3D-EPI (Reference)",
        "3D-EPI (Prisma)",
        "SSFP (Reference)",
        "SSFP (Prisma)"
    ]

    mni_registered_data_dirs = []
    mni_result_layouts = []
    for dataset_dir in dataset_dirs:
        mni_registered_data_dir = os.path.join(dataset_dir, "registered")
        mni_registered_data_dirs.append(mni_registered_data_dir)
        mni_result_layouts.append(BIDSLayout(mni_registered_data_dir, validate=False))


    sub_ses_run_dicts = []
    subjects = mni_result_layouts[0].get_subjects()

    subject_run_combinations = [
        dict(subject="phy001", run=1),
        dict(subject="phy001", run=2),
        dict(subject="phy002", run=None),
        dict(subject="phy003", run=None),
        dict(subject="phy004", run=None),
    ]

    # subject_run_combinations = [subject_run_combinations[0]]

    for subject_run_combination in subject_run_combinations:

        r2_maps = []
        for layout in mni_result_layouts:

            t2_map_files = layout.get(**subject_run_combination,
                                      space="MNI152",
                                      suffix="T2map",
                                      extension="nii.gz")
            print(subject_run_combination)
            print(t2_map_files)
            assert (len(t2_map_files) == 1)
            t2_map_file = t2_map_files[0]

            r2_map_files = layout.get(**subject_run_combination,
                                      space="MNI152",
                                      suffix="R2map",
                                      extension="nii.gz")
            assert (len(r2_map_files) == 1)
            r2_map_file = r2_map_files[0]

            sub_ses_run_dicts.append(dict(subject=subject_run_combination["subject"],
                                          run=subject_run_combination["run"]))

            r2_map_nib = nib.load(r2_map_file)
            r2_map = r2_map_nib.get_fdata()
            r2_map[brain_mask<=0] = np.nan
            r2_maps.append(r2_map)

        fig, axes = plt.subplots(3, len(r2_maps), figsize=(15, 9))  # 3 views x 5 maps

        # Loop through each volume (R2 map) and plot the three central slices
        for i, volume in enumerate(r2_maps):

            if i < 2:
                volume[volume>35] = 0

            if same_scaling:
                vmin, vmax = 0, 38
            else:
                vmin = np.percentile(volume[~np.isnan(volume)], 1)
                vmax = np.percentile(volume[~np.isnan(volume)], 99)

                non_nan_mask = ~np.isnan(volume)
                # volume = (volume-np.mean(volume[non_nan_mask]))/np.std(volume[non_nan_mask])
                volume = (volume - np.median(volume[non_nan_mask])) / scipy.stats.median_abs_deviation(
                    volume[non_nan_mask])

                vmin = -3
                vmax = 3

            sagittal, coronal, axial = get_central_slices(volume)


            label_txt = 'R_2 robust z-score'

            # Define colormap and color range
            cmap = 'coolwarm'  # You can change this to any colormap you prefer

            # Plot sagittal slice (first row)
            im1 = axes[0, i].imshow(np.rot90(sagittal), cmap=cmap, vmin=vmin,
                                    vmax=vmax)
            axes[0, i].set_title(dataset_names[i])
            axes[0, i].axis('off')  # Turn off axis labels
            cbar = fig.colorbar(im1, ax=axes[0, i], fraction=0.046,
                         pad=0.04)  # Add colorbar
            cbar.set_label(label_txt)

            # Plot coronal slice (second row)
            im2 = axes[1, i].imshow(np.rot90(coronal), cmap=cmap, vmin=vmin,
                                    vmax=vmax)
            axes[1, i].axis('off')  # Turn off axis labels
            cbar = fig.colorbar(im2, ax=axes[1, i], fraction=0.046,
                         pad=0.04)  # Add colorbar
            cbar.set_label(label_txt)

            # Plot axial slice (third row)
            im3 = axes[2, i].imshow(np.rot90(axial), cmap=cmap, vmin=vmin,
                                    vmax=vmax)
            axes[2, i].axis('off')  # Turn off axis labels
            cbar = fig.colorbar(im3, ax=axes[2, i], fraction=0.046,
                         pad=0.04)  # Add colorbar
            cbar.set_label(label_txt)

        # Add row titles for the views
        axes[0, 0].set_ylabel('Sagittal', fontsize=12)
        axes[1, 0].set_ylabel('Coronal', fontsize=12)
        axes[2, 0].set_ylabel('Axial', fontsize=12)

        subject_id = subject_run_combination['subject']
        if same_scaling:
            fig.suptitle(
                f"R2 Maps for Subject {subject_id} (same colorbar range)", fontsize=16)
        else:
            fig.suptitle(f"R2 Maps for Subject {subject_id} (scaled per scan)", fontsize=16)


        brain_shape = brain_mask.shape
        for ax in axes.flatten():
            # ax.set_xlim(0, max(brain_shape))  # Set x-limits
            # ax.set_ylim(0, max(brain_shape))  # Set y-limits
            ax.set_aspect('equal')  # Set equal aspect ratio

        # Adjust the layout
        plt.tight_layout()
        # Adjust the layout
        plt.tight_layout()

        # Save the plot to a file (e.g., PNG)
        subject_id = subject_run_combination['subject']
        run_id = subject_run_combination['run'] if subject_run_combination[
            'run'] else ''
        output_filename = f"subject_{subject_id}_{run_id}_r2_map.png"
        plt.savefig(os.path.join(out_dir, output_filename),
                    dpi=300)

        # Show the plot without blocking
        plt.pause(0.001)

        # Clear the figure to free memory for the next plot
        plt.clf()

