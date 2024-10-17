import argparse
import os
import multiprocessing
from nipype import Workflow, Node, Function
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.fsl import Info
from nipype.interfaces.ants import ApplyTransforms
from bids.layout import BIDSLayout
import nipype.pipeline.engine as pe
import nipype.interfaces.ants as ants
from nipype_utils import BidsOutputWriter
from utils.io import write_minimal_bids_dataset_description
from nipype.interfaces.utility import Select
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    out_dir = "../data/figures/qualitative/same_scaling"
    os.makedirs(out_dir, exist_ok=True)

    cortical_regions = [
        dict(label=7, name="Temporal Pole"),
        dict(label=33, name="Parahippocampal Gyrus, anterior division"),
        dict(label=1, name="Insular Cortex"),
        dict(label=6, name="Precentral Gyrus"),
        dict(label=16, name="Postcentral Gyrus"),
        dict(label=47, name="Occipital Pole"),
        dict(label=24, name="Frontal Medial Cortex"),
    ]

    subcortical_regions = [
        dict(label=[8, 18], name="Hippocampus"),
        dict(label=[5, 16], name="Putamen")
    ]

    brain_mask_file = "../data/templates/MNI152_T1_1mm_brain_mask.nii.gz"
    brain_mask = nib.load(brain_mask_file).get_fdata()

    dataset_dirs = [
        "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids_results_bad_b1",
        "/media/laurin/Data_share/Travel_Head_Study/London_Kings_Vida_3T_bids_results",
        "/media/laurin/Data_share/Travel_Head_Study/Hamburg_Prisma_3T_bids_results/fibu"
    ]
    dataset_names = [
        "3D-EPI (Reference)",
        "SSFP (Reference)",
        "SSFP (Prisma)",
    ]

    mni_registered_data_dirs = []
    mni_result_layouts = []
    for dataset_dir in dataset_dirs:
        mni_registered_data_dir = os.path.join(dataset_dir, "registeredMni")
        mni_registered_data_dirs.append(mni_registered_data_dir)

        print(f"Loading BIDS layout for dataset: {dataset_dir}")
        mni_result_layouts.append(
            BIDSLayout(mni_registered_data_dir, validate=False))

    sub_ses_run_dicts = []
    subjects = mni_result_layouts[0].get_subjects()

    subject_run_combinations = [
        dict(subject="phy001", run=1),
        dict(subject="phy002", run=None),
        dict(subject="phy003", run=None),
        dict(subject="phy004", run=None),
    ]

    region_data = np.empty(
        (len(mni_result_layouts), len(subject_run_combinations)), dtype=object)
    # sub_region_data = np.empty(
    #     (len(mni_result_layouts), len(subject_run_combinations)), dtype=object)

    # Start processing each site
    for site_index, layout in enumerate(mni_result_layouts):
        print(
            f"\nProcessing site {site_index + 1}/{len(mni_result_layouts)}: {dataset_names[site_index]}")

        # Start processing each subject/run combination
        for subject_run_index, subject_run_combination in enumerate(
                subject_run_combinations):
            print(
                f"  Collecting data for subject: {subject_run_combination['subject']}, run: {subject_run_combination.get('run', 'None')}")

            # Load probability segmentation and T2 map files
            cortical_probseg_file = "../data/atlases/HarvardOxford-cort-prob-1mm.nii.gz"
            subcortical_probseg_file = "../data/atlases/HarvardOxford-sub-prob-1mm.nii.gz"

            t2_map_files = layout.get(**subject_run_combination, suffix="T2Map",
                                      extension="nii.gz")
            assert (len(t2_map_files) == 1)
            t2_map_files = t2_map_files[0]

            print("    - Loading T2 map and probability segmentation files")

            # Load the files into memory
            t2_map_nib = nib.load(t2_map_files)
            t2_map = t2_map_nib.get_fdata()
            t2_map[brain_mask <= 0] = np.nan

            cortical_probseg_nib = nib.load(cortical_probseg_file)
            cortical_probseg = cortical_probseg_nib.get_fdata()

            subcortical_probseg_nib = nib.load(subcortical_probseg_file)
            subcortical_probseg = subcortical_probseg_nib.get_fdata()

            region_thres = 75

            # Extract T2 values for each cortical region
            region_dict = dict()
            for cortical_region in cortical_regions:
                roi_probseg = cortical_probseg[:, :, :,
                              cortical_region["label"]]
                roi_mask = roi_probseg > region_thres
                t2_map_roi = t2_map[roi_mask]
                t2_map_roi = t2_map_roi[~np.isnan(t2_map_roi)]
                region_dict[cortical_region["name"]] = t2_map_roi

            for subcortical_region in subcortical_regions:

                labels = subcortical_region["label"]
                roi_probseg = np.zeros(subcortical_probseg.shape[:3])
                for label in labels:
                    roi_probseg += subcortical_probseg[:, :, :,
                                  label]
                roi_mask = roi_probseg > region_thres
                t2_map_roi = t2_map[roi_mask]
                t2_map_roi = t2_map_roi[~np.isnan(t2_map_roi)]
                region_dict[subcortical_region["name"]] = t2_map_roi


            # Save region data
            region_data[site_index, subject_run_index] = region_dict
            print("    - T2 values extracted and saved for regions")

    print("\nData collection completed. Now pooling data across sites.")

    # Pool T2 values over all subjects for each site and region
    cortical_region_names = [region["name"] for region in cortical_regions]
    subcortical_region_names = [region["name"] for region in subcortical_regions]
    regions = cortical_region_names + subcortical_region_names

    pooled_region_data = {region: {site: [] for site in dataset_names} for
                          region in regions}

    for site_index, site_name in enumerate(dataset_names):
        for subject_run_index in range(len(subject_run_combinations)):
            region_dict = region_data[site_index, subject_run_index]
            if region_dict is not None:
                for region_name in regions:
                    pooled_region_data[region_name][site_name].extend(
                        region_dict[region_name])

    print("Data pooling completed. Now generating boxplots.")

    # Create a 1xN grid of boxplots where N is the number of cortical regions
    num_regions = len(regions)
    fig, axes = plt.subplots(3, 3, figsize=(num_regions * 5, 5),
                             sharey=True)

    # Loop over each cortical region and create boxplots
    ax_flat = axes.flatten()
    for i, region_name in enumerate(regions):
        region_values = pooled_region_data[region_name]

        # # Prepare data for the boxplot (grouped by site)
        # data = [region_values[site] for site in dataset_names]
        #
        # # Create a boxplot for the region
        # ax_flat[i].boxplot(data, labels=dataset_names)
        # ax_flat[i].set_title(region_name)
        # ax_flat[i].set_xlabel("Site")
        # ax_flat[i].set_ylim([0, 0.2])

        # Calculate mean and standard deviation for each site
        means = [np.mean(region_values[site]) for site in dataset_names]
        stds = [np.std(region_values[site]) for site in dataset_names]

        # X-axis for the sites
        x = np.arange(len(dataset_names))

        # Create a plot for the region (mean and std)
        ax_flat[i].errorbar(x, means, yerr=stds, fmt='o', capsize=5, capthick=2)
        ax_flat[i].set_title(region_name)
        ax_flat[i].set_xticks(x)
        ax_flat[i].set_xticklabels(dataset_names)
        ax_flat[i].set_xlabel("Site")

        if i == 0:
            ax_flat[i].set_ylabel(
                "T2 [ms]")  # Only add y-axis label to the first plot

    # Adjust the layout
    plt.tight_layout()

    # Save or show the plot
    # plt.savefig(os.path.join(out_dir, "t2_region_comparison_boxplot.png"), dpi=300)
    plt.show()

    print("Boxplots generated and saved.")
