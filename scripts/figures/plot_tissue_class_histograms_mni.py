import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
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

    out_dir = "../../data/figures/quantitative"
    os.makedirs(out_dir, exist_ok=True)

    mni_registered_data_dirs = []
    subject_registered_data_dirs = []
    processed_data_dirs = []
    segmented_data_dirs = []
    mni_result_layouts = []
    for dataset_dir in results_dataset_dirs:
        subject_registered_data_dir = os.path.join(dataset_dir,
                                                   "registered")
        subject_registered_data_dirs.append(subject_registered_data_dir)
        processed_data_dir = os.path.join(dataset_dir, "processed")
        processed_data_dirs.append(processed_data_dir)

        print(f"Loading BIDS layout for dataset: {dataset_dir}")
        mni_result_layouts.append(
            BIDSLayout(processed_data_dir, derivatives=[
                subject_registered_data_dir],
                       validate=False))

    sub_ses_run_dicts = []
    subjects = mni_result_layouts[0].get_subjects()

    subject_run_combinations = [
        dict(subject="phy001", run=1),
        dict(subject="phy001", run=2),
        dict(subject="phy002", run=None),
        dict(subject="phy003", run=None),
        dict(subject="phy004", run=None),
    ]

    # Define the figure with subplots arranged by subjects (rows) and datasets (columns)
    fig, axes = plt.subplots(len(subject_run_combinations),
                             len(dataset_names), figsize=(9, 9),
                             sharex=False, sharey=True)

    # Start processing each site
    for site_index, layout in enumerate(mni_result_layouts):
        print(
            f"\nProcessing site {site_index + 1}/{len(mni_result_layouts)}: {dataset_names[site_index]}")

        # Start processing each subject/run combination
        for subject_run_index, subject_run_combination in enumerate(
                subject_run_combinations):
            print(
                f"  Collecting data for subject: {subject_run_combination['subject']}, run: {subject_run_combination.get('run', 'None')}")

            wm_probseg_files = layout.get(**subject_run_combination,
                                          label="wm",
                                          suffix="probseg",
                                          space="subject",
                                          extension="nii.gz")
            assert (len(wm_probseg_files) == 1)
            wm_probseg_file = wm_probseg_files[0]

            gm_probseg_files = layout.get(**subject_run_combination,
                                          label="gm",
                                          suffix="probseg",
                                          space="subject",
                                          extension="nii.gz")
            assert (len(gm_probseg_files) == 1)
            gm_probseg_file = gm_probseg_files[0]

            r2_map_files = layout.get(**subject_run_combination,
                                      suffix="R2map",
                                      extension="nii.gz",
                                      space=None)
            assert (len(r2_map_files) == 1)
            r2_map_file = r2_map_files[0]

            print("    - Loading R2 map and probability segmentation files")

            # Load the files into memory
            r2_map_nib = nib.load(r2_map_file)
            r2_map = r2_map_nib.get_fdata()

            wm_probseg_nib = nib.load(wm_probseg_file)
            wm_probseg = wm_probseg_nib.get_fdata()

            gm_probseg_nib = nib.load(gm_probseg_file)
            gm_probseg = gm_probseg_nib.get_fdata()

            tissue_threshold = 0.95
            wm_mask = wm_probseg >= tissue_threshold
            gm_mask = gm_probseg >= tissue_threshold

            wm_values = r2_map[wm_mask]
            gm_values = r2_map[gm_mask]

            # Get the axis for the current subject and site (dataset)
            ax = axes[subject_run_index, site_index]

            # Plot histograms for WM, GM, and CSF
            bins = np.arange(0, 60, 0.2)
            # Plot normalized histograms for WM, GM, and CSF
            ax.hist(wm_values, bins=bins, alpha=0.5, color='red', label='WM',
                    density=True)
            ax.hist(gm_values, bins=bins, alpha=0.5, color='blue', label='GM',
                    density=True)

            # Set the title for each dataset column in the top row
            if subject_run_index == 0:
                ax.set_title(dataset_names[site_index])

            # Set labels for the first column in each row
            if site_index == 0:
                subject_run_txt = f"sub-{subject_run_combination['subject']}"
                if subject_run_combination["run"] is not None:
                    subject_run_txt += f"_run-{subject_run_combination['run']}"
                ax.set_ylabel(f"{subject_run_txt}\n\nFrequency")

            ax.set_xlabel("$R_2$ values [1/$s$]")
            ax.legend(loc='upper right')

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    output_filename = "r2_tissue_class_histograms.png"
    plt.savefig(os.path.join(out_dir, output_filename),
                dpi=300)
