import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
from bids.layout import BIDSLayout
from nipype.interfaces import fsl

if __name__ == "__main__":

    brain_mask_file = fsl.Info.standard_image('MNI152_T1_1mm_brain_mask.nii.gz')
    brain_mask = nib.load(brain_mask_file).get_fdata()

    dataset_names = [
        "MPR\n(reference site)",
        "MPR\n(target site)",
        "CSMT-JSR\n(reference site)",
        "JSR\n(target site)"
    ]
    results_dir = "../../data/results"
    results_dataset_dirs = [os.path.join(results_dir, d) for d in [
        "dzne_3depi",
        "uke_3depi",
        "kings_ssfp_spgr",
        "uke_ssfp_spgr"
    ]]

    out_dir = "../../data/figures/qualitative"
    os.makedirs(out_dir, exist_ok=True)

    mni_registered_data_dirs = []
    mni_result_layouts = []
    for results_dataset_dir in results_dataset_dirs:
        mni_registered_data_dir = os.path.join(results_dataset_dir, "registered")
        mni_registered_data_dirs.append(mni_registered_data_dir)
        mni_result_layouts.append(
            BIDSLayout(mni_registered_data_dir, validate=False))

    sub_ses_run_dicts = []
    subjects = mni_result_layouts[0].get_subjects()

    subject_run_combinations = [
        dict(subject="phy001", run=1),
        dict(subject="phy001", run=2),
        dict(subject="phy002", run=None),
        dict(subject="phy003", run=None),
        dict(subject="phy004", run=None),
    ]

    for subject_run_combination in subject_run_combinations:

        # get R2 maps across sites
        r2_maps = []
        for layout in mni_result_layouts:
            r2_map_files = layout.get(**subject_run_combination,
                                      space="MNI152",
                                      suffix="R2map",
                                      extension="nii.gz")
            assert (len(r2_map_files) == 1)
            r2_map_file = r2_map_files[0]

            sub_ses_run_dicts.append(
                dict(subject=subject_run_combination["subject"],
                     run=subject_run_combination["run"]))

            r2_map_nib = nib.load(r2_map_file)
            r2_map = r2_map_nib.get_fdata()
            r2_map[brain_mask <= 0] = np.nan
            r2_maps.append(r2_map)

        fig, axes = plt.subplots(3, len(r2_maps),
                                 figsize=(10, 9))  # 3 views x 5 maps

        # Show 2D views per of R2 per site
        for site_index, volume in enumerate(r2_maps):
            # compute z score
            non_nan_mask = ~np.isnan(volume)
            z_score = (volume - np.median(
                volume[non_nan_mask])) / scipy.stats.median_abs_deviation(
                volume[non_nan_mask])

            # extract nice 2D views
            sagittal_view = z_score[87, :, :]
            coronal_view = z_score[:, 80, :]
            axial_view = z_score[:, :, 85]

            # plot 2d views
            views = [sagittal_view, coronal_view, axial_view]
            for view_index, view in enumerate(views):
                ax = axes[view_index, site_index]
                im1 = ax.imshow(np.rot90(view),
                                cmap='coolwarm',
                                vmin=-3,
                                vmax=3)
                ax.axis('off')
                ax.set_aspect('equal')

                # add column title
                if view_index == 0:
                    ax.set_title(dataset_names[site_index])

        # add single colorbar on top
        cbar_ax = fig.add_axes(
            [0.1, 0.95, 0.8, 0.03])  # Position: [left, bottom, width, height]
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("$R_2$ $z$-score [n.u.]")

        # add row labels
        fig.text(0.01, 0.7, 'Sagittal', va='center', ha='center',
                 fontsize=12, rotation=90)
        fig.text(0.01, 0.42, 'Coronal', va='center', ha='center',
                 fontsize=12, rotation=90)
        fig.text(0.01, 0.15, 'Axial', va='center', ha='center',
                 fontsize=12, rotation=90)

        # tighten layout
        plt.tight_layout(rect=[0.01, 0, 1, 0.87], pad=0.0)
        plt.subplots_adjust(wspace=-0.12, hspace=-0.03, top=0.85)

        # Save the plot
        subject_id = subject_run_combination['subject']
        run_txt = f"_run-{subject_run_combination['run']}" if \
        subject_run_combination[
            'run'] else ''
        output_filename = f"subject-{subject_id}{run_txt}_r2_zscore_map.png"
        plt.savefig(os.path.join(out_dir, output_filename),
                    dpi=300)
        plt.close(fig)