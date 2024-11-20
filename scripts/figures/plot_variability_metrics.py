import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
from bids.layout import BIDSLayout
# Create custom legend handles
from matplotlib.patches import Patch

if __name__ == "__main__":

    out_dir = "../../data/figures/quantitative"
    os.makedirs(out_dir, exist_ok=True)

    # Filepath for storing the data
    pickle_file = os.path.join(out_dir, "region_data.pkl")

    cortical_regions = [
        dict(label=7, name="Temporal Pole"),
        dict(label=33, name="Parahippocampal Gyrus anterior"),
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
        dict(subject="phy002", run=None),
        dict(subject="phy003", run=None),
        dict(subject="phy004", run=None),
    ]

    region_data = np.empty(
        (len(mni_result_layouts), len(subject_run_combinations)),
        dtype=object)

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
            cortical_probseg_files = layout.get(**subject_run_combination,
                                                label="cortical",
                                                suffix="probseg",
                                                space="subject",
                                                extension="nii.gz")
            assert (len(cortical_probseg_files) == 1)
            cortical_probseg_file = cortical_probseg_files[0]

            subcortical_probseg_files = layout.get(
                **subject_run_combination,
                label="subcortical",
                suffix="probseg",
                space="subject",
                extension="nii.gz")
            assert (len(subcortical_probseg_files) == 1)
            subcortical_probseg_file = subcortical_probseg_files[0]

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

            cortical_probseg_nib = nib.load(cortical_probseg_file)
            cortical_probseg = cortical_probseg_nib.get_fdata()

            subcortical_probseg_nib = nib.load(subcortical_probseg_file)
            subcortical_probseg = subcortical_probseg_nib.get_fdata()

            gm_mask_nib = nib.load(gm_probseg_file)
            gm_mask = gm_mask_nib.get_fdata() >= 0.95
            atlas_roi_thres = 0.3
            r2_thres = 100

            # Extract T2 values for each cortical region
            region_dict = dict()

            for cortical_region in cortical_regions:
                roi_probseg = cortical_probseg[:, :, :,
                              cortical_region["label"]]
                roi_mask = roi_probseg / 100
                roi_mask[roi_mask < atlas_roi_thres] = 0
                roi_mask = np.logical_and(roi_mask, gm_mask)
                roi_mask = roi_mask.astype(bool)
                r2_map_roi = r2_map[roi_mask]
                r2_map_roi = r2_map_roi[r2_map_roi < r2_thres]
                region_dict[cortical_region["name"]] = r2_map_roi

            for subcortical_region in subcortical_regions:
                labels = subcortical_region["label"]

                left_label = labels[0]
                roi_probseg = subcortical_probseg[:, :, :,
                              left_label]
                roi_mask = roi_probseg / 100
                roi_mask[roi_mask < atlas_roi_thres] = 0
                roi_mask = np.logical_and(roi_mask, gm_mask)
                roi_mask = roi_mask.astype(bool)
                r2_map_roi = r2_map[roi_mask]
                r2_map_roi = r2_map_roi[r2_map_roi < r2_thres]
                r2_map_roi = r2_map_roi[~np.isnan(r2_map_roi)]

                right_label = labels[1]
                roi_probseg = subcortical_probseg[:, :, :,
                              right_label]
                roi_mask = roi_probseg / 100
                roi_mask[roi_mask < atlas_roi_thres] = 0
                roi_mask = np.logical_and(roi_mask, gm_mask)
                roi_mask = roi_mask.astype(bool)
                r2_map_roi = r2_map[roi_mask]
                r2_map_roi = r2_map_roi[r2_map_roi < r2_thres]
                r2_map_roi = r2_map_roi[~np.isnan(r2_map_roi)]

                roi_probseg = np.zeros(subcortical_probseg.shape[:3])
                for label in labels:
                    roi_probseg += subcortical_probseg[:, :, :,
                                   label]
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(bool)
                roi_mask = roi_probseg / 100
                roi_mask[roi_mask < atlas_roi_thres] = 0
                roi_mask = np.logical_and(roi_mask, gm_mask)
                roi_mask = roi_mask.astype(bool)
                roi_mask = roi_mask.astype(bool)
                r2_map_roi = r2_map[roi_mask]
                r2_map_roi = r2_map_roi[r2_map_roi < r2_thres]
                r2_map_roi = r2_map_roi[~np.isnan(r2_map_roi)]
                region_dict[subcortical_region["name"]] = r2_map_roi

            # Save region data
            region_data[site_index, subject_run_index] = region_dict

            print("    - R2 values extracted and saved for regions")

    print("\nData collection completed. Now pooling data across sites.")

    # Pool T2 values over all subjects for each site and region
    region_names = [region["name"] for region in cortical_regions] + [
        region["name"] for region in subcortical_regions]

    print("Data pooling completed. Now generating boxplots.")


    # Number of sites and regions
    num_sites = len(dataset_names)
    num_regions = len(region_names)

    means_per_site_subject_region = np.zeros(
        (region_data.shape[0], region_data.shape[1], len(region_names)),
        dtype=float)

    for site_idx, site_name in enumerate(dataset_names):
        for region_index, region_name in enumerate(region_names):
            for subject_run_index in range(0, region_data.shape[1]):
                means_per_site_subject_region[
                    site_idx, subject_run_index, region_index] = np.mean(
                    region_data[site_idx, subject_run_index][region_name])

    inter_subject_var = (np.percentile(means_per_site_subject_region, 75, 1) - np.percentile(means_per_site_subject_region, 25, 1)) / np.median(
        means_per_site_subject_region, 1)
    inter_region_var =  (np.percentile(means_per_site_subject_region, 75, 2) - np.percentile(means_per_site_subject_region, 25, 2)) / np.median(
        means_per_site_subject_region, 2)


    vals_per_site_region_median = np.zeros(
        (region_data.shape[0], len(region_names)),
        dtype=float)
    vals_per_site_region_mad = np.zeros(
        (region_data.shape[0], len(region_names)),
        dtype=float)
    vals_per_site_region_iqr = np.zeros(
        (region_data.shape[0], len(region_names)),
        dtype=float)
    for site_idx, site_name in enumerate(dataset_names):
        for region_index, region_name in enumerate(region_names):
            vals = []
            for subject_run_index in range(0, region_data.shape[1]):
                vals.extend(region_data[site_idx, subject_run_index][region_name])
            vals_per_site_region_median[site_idx, region_index] = np.median(vals)
            vals_per_site_region_mad[site_idx, region_index] = scipy.stats.median_abs_deviation(vals)
            vals_per_site_region_iqr[site_idx, region_index] = np.percentile(vals,75)-np.percentile(vals,25)

    intra_region_var = vals_per_site_region_iqr/vals_per_site_region_median

    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']

    # Create a 2xN grid where the first row is for boxplots and the second row is for inter-hemispheric plots
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    inter_subject_var_ax = axes[2]
    inter_region_var_ax = axes[0]
    intra_region_var_ax = axes[1]

    inter_subject_var_ax.bar(range(0, len(dataset_names)),
                100*np.mean(inter_subject_var, 1),
                yerr=100*np.std(inter_subject_var, 1),
                edgecolor='black', color=colors)
    inter_subject_var_ax.set_xlabel("Site")
    inter_subject_var_ax.set_ylabel("Inter-subject CoV [%]")
    inter_subject_var_ax.set_xticks([])
    inter_subject_var_ax.set_xticklabels([])
    inter_subject_var_ax.set_ylim([0, 35])

    inter_region_var_ax.bar(range(0, len(dataset_names)),
                100*np.mean(inter_region_var, 1),
                yerr=100*np.std(inter_region_var, 1),
                edgecolor='black', color=colors)
    inter_region_var_ax.set_xlabel("Site")
    inter_region_var_ax.set_ylabel("Inter-region CoV [%]")

    inter_region_var_ax.set_xticks([])
    inter_region_var_ax.set_xticklabels([])
    inter_region_var_ax.set_ylim([0, 35])

    intra_region_var_ax.bar(range(0, len(dataset_names)),
                100*np.mean(intra_region_var, 1),
                yerr=100*np.std(intra_region_var, 1),
                edgecolor='black', color=colors)
    intra_region_var_ax.set_xlabel("Site")
    intra_region_var_ax.set_ylabel("Intra-region CoV [%]")
    intra_region_var_ax.set_xticks([])
    intra_region_var_ax.set_xticklabels([])
    intra_region_var_ax.set_ylim([0, 35])


    legend_elements = []
    for dataset_name, color in zip(dataset_names, colors):
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=dataset_name))

    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.03), frameon=False)

    # Save or show the plot
    plt.tight_layout(
        rect=[0.01, 0, 1, 0.85], pad=0.1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(os.path.join(out_dir, "r2_variability_metrics.png"),
                dpi=300)
