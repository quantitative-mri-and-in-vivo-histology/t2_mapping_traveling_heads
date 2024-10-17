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
    subject_registered_data_dirs = []
    processed_data_dirs = []
    mni_result_layouts = []
    for dataset_dir in dataset_dirs:
        subject_registered_data_dir = os.path.join(dataset_dir, "registeredSubject")
        subject_registered_data_dirs.append(subject_registered_data_dir)
        processed_data_dir = os.path.join(dataset_dir, "processed")
        processed_data_dirs.append(processed_data_dir)

        print(f"Loading BIDS layout for dataset: {dataset_dir}")
        mni_result_layouts.append(BIDSLayout(processed_data_dir, derivatives=[subject_registered_data_dir], validate=False))

    sub_ses_run_dicts = []
    subjects = mni_result_layouts[0].get_subjects()

    subject_run_combinations = [
        dict(subject="phy001", run=1),
        dict(subject="phy002", run=None),
        dict(subject="phy003", run=None),
        dict(subject="phy004", run=None),
    ]

    region_data = np.empty((len(mni_result_layouts), len(subject_run_combinations)), dtype=object)

    # Start processing each site
    for site_index, layout in enumerate(mni_result_layouts):
        print(f"\nProcessing site {site_index + 1}/{len(mni_result_layouts)}: {dataset_names[site_index]}")

        # Start processing each subject/run combination
        for subject_run_index, subject_run_combination in enumerate(subject_run_combinations):
            print(f"  Collecting data for subject: {subject_run_combination['subject']}, run: {subject_run_combination.get('run', 'None')}")

            # Load probability segmentation and T2 map files
            cortical_probseg_files = layout.get(**subject_run_combination, desc="cortical", suffix="probseg", extension="nii.gz")
            assert (len(cortical_probseg_files) == 1)
            cortical_probseg_file = cortical_probseg_files[0]

            subcortical_probseg_files = layout.get(**subject_run_combination, desc="subcortical", suffix="probseg", extension="nii.gz")
            assert (len(subcortical_probseg_files) == 1)
            subcortical_probseg_file = subcortical_probseg_files[0]

            t2_map_files = layout.get(**subject_run_combination, suffix="T2map", extension="nii.gz")
            assert (len(t2_map_files) == 1)
            t2_map_files = t2_map_files[0]

            print("    - Loading T2 map and probability segmentation files")

            # Load the files into memory
            t2_map_nib = nib.load(t2_map_files)
            t2_map = t2_map_nib.get_fdata()

            cortical_probseg_nib = nib.load(cortical_probseg_file)
            cortical_probseg = cortical_probseg_nib.get_fdata()

            subcortical_probseg_nib = nib.load(subcortical_probseg_file)
            subcortical_probseg = subcortical_probseg_nib.get_fdata()

            # Extract T2 values for each cortical region
            region_dict = dict()
            for cortical_region in cortical_regions:
                roi_probseg = cortical_probseg[:, :, :, cortical_region["label"]]
                roi_mask = roi_probseg > 50
                t2_map_roi = t2_map[roi_mask]
                region_dict[cortical_region["name"]] = t2_map_roi

            # Save region data
            region_data[site_index, subject_run_index] = region_dict
            print("    - T2 values extracted and saved for regions")

    print("\nData collection completed. Now pooling data across sites.")

    # Pool T2 values over all subjects for each site and region
    cortical_region_names = [region["name"] for region in cortical_regions]
    pooled_region_data = {region: {site: [] for site in dataset_names} for region in cortical_region_names}

    for site_index, site_name in enumerate(dataset_names):
        for subject_run_index in range(len(subject_run_combinations)):
            region_dict = region_data[site_index, subject_run_index]
            if region_dict is not None:
                for region_name in cortical_region_names:
                    pooled_region_data[region_name][site_name].extend(region_dict[region_name])

    print("Data pooling completed. Now generating boxplots.")

    # Create a 1xN grid of boxplots where N is the number of cortical regions
    num_regions = len(cortical_region_names)
    fig, axes = plt.subplots(1, num_regions, figsize=(num_regions * 5, 5), sharey=True)

    # Loop over each cortical region and create boxplots
    for i, region_name in enumerate(cortical_region_names):
        region_values = pooled_region_data[region_name]

        # Prepare data for the boxplot (grouped by site)
        data = [region_values[site] for site in dataset_names]

        # Create a boxplot for the region
        axes[i].boxplot(data, labels=dataset_names)
        axes[i].set_title(region_name)
        axes[i].set_xlabel("Site")
        if i == 0:
            axes[i].set_ylabel("T2 [ms]")  # Only add y-axis label to the first plot

    # Adjust the layout
    plt.tight_layout()

    # Save or show the plot
    # plt.savefig(os.path.join(out_dir, "t2_region_comparison_boxplot.png"), dpi=300)
    plt.show()

    print("Boxplots generated and saved.")
