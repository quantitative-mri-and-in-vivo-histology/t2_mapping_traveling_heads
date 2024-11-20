import os
import pickle

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from bids.layout import BIDSLayout

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

    regenerate_data = True

    if regenerate_data:

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
            dict(subject="phy002", run=None),
            dict(subject="phy003", run=None),
            dict(subject="phy004", run=None),
        ]

        region_data = np.empty(
            (len(mni_result_layouts), len(subject_run_combinations)),
            dtype=object)
        region_data_left = np.empty(
            (len(mni_result_layouts), len(subject_run_combinations)),
            dtype=object)
        region_data_right = np.empty(
            (len(mni_result_layouts), len(subject_run_combinations)),
            dtype=object)

        region_subject_mean_data = np.empty(
            (len(mni_result_layouts), len(subject_run_combinations)),
            dtype=object)
        region_subject_mean_data_left = np.empty(
            (len(mni_result_layouts), len(subject_run_combinations)),
            dtype=object)
        region_subject_mean_data_right = np.empty(
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

                # Load probability segmentation and R2 map files
                cortical_probseg_files = layout.get(**subject_run_combination,
                                                    label="cortical",
                                                    suffix="probseg",
                                                    space="subject",
                                                    extension="nii.gz")
                assert (len(cortical_probseg_files) == 1)
                cortical_probseg_file = cortical_probseg_files[0]

                cortical_left_probseg_files = layout.get(
                    **subject_run_combination,
                    label="corticalLeft",
                    suffix="probseg",
                    space="subject",
                    extension="nii.gz")
                print(subject_run_combination)
                print(cortical_left_probseg_files)
                assert (len(cortical_left_probseg_files) == 1)
                cortical_left_probseg_file = cortical_left_probseg_files[0]

                cortical_right_probseg_files = layout.get(
                    **subject_run_combination,
                    label="corticalRight",
                    suffix="probseg",
                    space="subject",
                    extension="nii.gz")
                assert (len(cortical_right_probseg_files) == 1)
                cortical_right_probseg_file = cortical_right_probseg_files[0]

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

                cortical_left_probseg_nib = nib.load(cortical_left_probseg_file)
                cortical_left_probseg = cortical_left_probseg_nib.get_fdata()

                cortical_right_probseg_nib = nib.load(
                    cortical_right_probseg_file)
                cortical_right_probseg = cortical_right_probseg_nib.get_fdata()

                subcortical_probseg_nib = nib.load(subcortical_probseg_file)
                subcortical_probseg = subcortical_probseg_nib.get_fdata()

                gm_mask_nib = nib.load(gm_probseg_file)
                gm_mask = gm_mask_nib.get_fdata() >= 0.95
                atlas_roi_thres = 0.3
                r2_thres = 100

                # Extract R2 values for each cortical region
                region_dict = dict()
                region_dict_left = dict()
                region_dict_right = dict()

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

                    roi_probseg = cortical_left_probseg[:, :, :,
                                  cortical_region["label"]]
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(bool)
                    r2_map_roi = r2_map[roi_mask]
                    r2_map_roi = r2_map_roi[r2_map_roi < r2_thres]
                    region_dict_left[cortical_region["name"]] = r2_map_roi

                    roi_probseg = cortical_right_probseg[:, :, :,
                                  cortical_region["label"]]
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(bool)
                    r2_map_roi = r2_map[roi_mask]
                    r2_map_roi = r2_map_roi[r2_map_roi < r2_thres]
                    region_dict_right[cortical_region["name"]] = r2_map_roi

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
                    region_dict_left[subcortical_region["name"]] = r2_map_roi

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
                    region_dict_right[subcortical_region["name"]] = r2_map_roi

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
                region_data_left[
                    site_index, subject_run_index] = region_dict_left
                region_data_right[
                    site_index, subject_run_index] = region_dict_right

                print("    - R2 values extracted and saved for regions")

        print("\nData collection completed. Now pooling data across sites.")

        # Pool R2 values over all subjects for each site and region
        region_names = [region["name"] for region in cortical_regions] + [
            region["name"] for region in subcortical_regions]

        # Save the results to a pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                "region_data": region_data,
                "region_data_left": region_data_left,
                "region_data_right": region_data_right,
                "region_names": region_names,
                "dataset_names": dataset_names
            }, f)

        print(f"Data collection results saved to {pickle_file}.")

    else:
        # Load the results from the pickle file
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        # Extract variables from the loaded data
        region_data = data["region_data"]
        region_data_left = data["region_data_left"]
        region_data_right = data["region_data_right"]
        region_names = data["region_names"]
        dataset_names = data["dataset_names"]

        print("Data loaded successfully.")

    # Number of sites and regions
    num_sites = len(dataset_names)
    num_regions = len(region_names)

    # Generate a list of colors for each region (use any colormap you like)
    cmap = plt.get_cmap("tab10")  # You can choose another colormap if needed
    region_colors = [cmap(i % 10) for i in
                     range(num_regions)]  # Use the same color for each region
    region_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#bcbd22',  # Yellow-green
        '#17becf'  # Cyan
    ]

    # Create a 2xN grid where the first row is for boxplots and the second row is for inter-hemispheric plots
    fig, axes = plt.subplots(2, num_sites, figsize=(9, 6), sharey='row')

    for site_idx, site_name in enumerate(dataset_names):

        # get values per region pooled over all subjects
        region_values_across_subjects = []
        for region_index, region_name in enumerate(region_names):
            region_values_across_subjects_roi = []
            for subject_index in range(0, region_data.shape[1]):
                region_values_across_subjects_roi.extend(
                    region_data[site_idx][subject_index][region_name])
            region_values_across_subjects.append(
                region_values_across_subjects_roi)

        flierprops = dict(marker='.', markerfacecolor='black', markersize=1,
                          linestyle='none')

        # Create a boxplot for this site
        bplot = axes[0, site_idx].boxplot(region_values_across_subjects,
                                          labels=region_names,
                                          flierprops=flierprops,
                                          patch_artist=True)
        for median in bplot['medians']:
            median.set_color('black')

        # Color each boxplot according to the region color
        for patch, color in zip(bplot['boxes'], region_colors):
            patch.set_facecolor(color)

        # Add titles and labels
        axes[0, site_idx].set_title(f"{site_name}", fontsize=10)
        axes[0, site_idx].set_xticks([])
        axes[0, site_idx].set_xticklabels([])
        axes[0, site_idx].set_ylim([0, 40])  # Adjust the y-axis limit as needed
        axes[0, site_idx].set_xlim([0.5, len(region_names) + 0.5])
        axes[0, site_idx].set_xlabel("ROI")
        if site_idx == 0:
            axes[0, site_idx].set_ylabel("$R_2$ [1/$s$]")

        # compute get hemispheric means per region and subject
        region_values_across_subject_means_left = np.zeros(
            (len(region_names), region_data.shape[1]))
        region_values_across_subject_means_right = np.zeros(
            (len(region_names), region_data.shape[1]))
        for region_index, region_name in enumerate(region_names):
            for subject_index in range(0, region_data.shape[1]):
                region_values_across_subject_means_left[
                    region_index, subject_index] = \
                    np.mean(
                        region_data_left[site_idx][subject_index][region_name])
                region_values_across_subject_means_right[
                    region_index, subject_index] = \
                    np.mean(
                        region_data_right[site_idx][subject_index][region_name])

        # compute mean and std of interhemispheric differences
        interhemishperhic_diffs = region_values_across_subject_means_left - region_values_across_subject_means_right
        hemisphere_differences_per_region_mean = np.mean(
            interhemishperhic_diffs, 1)
        hemisphere_differences_per_region_std = np.std(interhemishperhic_diffs,
                                                       1)

        for i, diff in enumerate(hemisphere_differences_per_region_mean):
            axes[1, site_idx].barh(i,
                                   hemisphere_differences_per_region_mean[i],
                                   color=region_colors[i],
                                   edgecolor='black')
            axes[1, site_idx].errorbar(
                hemisphere_differences_per_region_mean[i], i,
                xerr=hemisphere_differences_per_region_std[i], fmt='none',
                ecolor='black',
                capsize=3)

        # Add labels and grid
        axes[1, site_idx].set_yticks([])
        axes[1, site_idx].set_yticklabels([])
        axes[1, site_idx].set_ylim([len(region_names) - 0.5, -0.5])
        axes[1, site_idx].set_xlim([-4, 4])
        axes[1, site_idx].axvline(0, color='black',
                                  linewidth=0.5)  # Vertical line at zero
        axes[1, site_idx].set_xlabel(
            "Interhemispheric (left-right)\n $R_2$ difference [1/s]")
        if site_idx == 0:
            axes[1, site_idx].set_ylabel("ROI")

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "r2_gm_roi_analysis.png"), dpi=300)
