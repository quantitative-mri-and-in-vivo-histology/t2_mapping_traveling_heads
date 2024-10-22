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
from requests import session
import pickle

from utils.io import write_minimal_bids_dataset_description
from nipype.interfaces.utility import Select
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    out_dir = "../../data/figures/qualitative"
    os.makedirs(out_dir, exist_ok=True)

    # Filepath for storing the data
    pickle_file = os.path.join(out_dir, "region_data_collection.pkl")

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

    # brain_mask_file = "../data/templates/MNI152_T1_1mm_brain_mask.nii.gz"
    # brain_mask = nib.load(brain_mask_file).get_fdata()

    # dataset_dirs = [
    #     "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids_results_bad_b1",
    #     "/media/laurin/Data_share/Travel_Head_Study/London_Kings_Vida_3T_bids_results",
    #     "/media/laurin/Data_share/Travel_Head_Study/Hamburg_Prisma_3T_bids_results/fibu"
    # ]

    # dataset_dirs = [
    #     "/media/laurin/Storage/Travel_Heads_Study/clean/results/Bonn_Skyra_3T_LowRes_bad_b1",
    #     "/media/laurin/Storage/Travel_Heads_Study/clean/results/London_Kings_Vida_3T",
    #     "/media/laurin/Storage/Travel_Heads_Study/clean/results/Hamburg_Prisma_3T_ssfp"
    # ]

    dataset_dirs = [
        "/media/laurin/Elements/Travel_Head_Study/clean/results/Bonn_Skyra_3T_LowRes_bad_b1",
        "/media/laurin/Elements/Travel_Head_Study/clean/results/London_Kings_Vida_3T",
        "/media/laurin/Elements/Travel_Head_Study/clean/results/Hamburg_Prisma_3T_ssfp"
    ]

    dataset_names = [
        "3D-EPI (Reference)",
        "SSFP (Reference)",
        "SSFP (Prisma)",
    ]

    regenerate_data = False

    if regenerate_data:

        mni_registered_data_dirs = []
        subject_registered_data_dirs = []
        processed_data_dirs = []
        segmented_data_dirs = []
        mni_result_layouts = []
        for dataset_dir in dataset_dirs:
            subject_registered_data_dir = os.path.join(dataset_dir,
                                                       "registeredSubject")
            subject_registered_data_dirs.append(subject_registered_data_dir)
            segmented_data_dir = os.path.join(dataset_dir, "segmented")
            segmented_data_dirs.append(segmented_data_dir)
            processed_data_dir = os.path.join(dataset_dir, "processed")
            processed_data_dirs.append(processed_data_dir)

            print(f"Loading BIDS layout for dataset: {dataset_dir}")
            mni_result_layouts.append(
                BIDSLayout(processed_data_dir, derivatives=[
                    subject_registered_data_dir, segmented_data_dir],
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

                # Load probability segmentation and T2 map files
                cortical_probseg_files = layout.get(**subject_run_combination,
                                                    desc="cortical",
                                                    suffix="probseg",
                                                    extension="nii.gz")
                assert (len(cortical_probseg_files) == 1)
                cortical_probseg_file = cortical_probseg_files[0]

                cortical_left_probseg_files = layout.get(
                    **subject_run_combination,
                    desc="corticalLeft",
                    suffix="probseg",
                    extension="nii.gz")
                print(subject_run_combination)
                print(cortical_left_probseg_files)
                assert (len(cortical_left_probseg_files) == 1)
                cortical_left_probseg_file = cortical_left_probseg_files[0]

                cortical_right_probseg_files = layout.get(
                    **subject_run_combination,
                    desc="corticalRight",
                    suffix="probseg",
                    extension="nii.gz")
                assert (len(cortical_right_probseg_files) == 1)
                cortical_right_probseg_file = cortical_right_probseg_files[0]

                subcortical_probseg_files = layout.get(
                    **subject_run_combination,
                    desc="subcortical",
                    suffix="probseg",
                    extension="nii.gz")
                assert (len(subcortical_probseg_files) == 1)
                subcortical_probseg_file = subcortical_probseg_files[0]

                gm_probseg_files = layout.get(**subject_run_combination,
                                              desc="gmPosterior",
                                              suffix="probseg",
                                              extension="nii.gz")
                assert (len(gm_probseg_files) == 1)
                gm_probseg_file = gm_probseg_files[0]

                t2_map_files = layout.get(**subject_run_combination,
                                          suffix="R2map",
                                          extension="nii.gz")
                assert (len(t2_map_files) == 1)
                t2_map_files = t2_map_files[0]

                print("    - Loading T2 map and probability segmentation files")

                # Load the files into memory
                t2_map_nib = nib.load(t2_map_files)
                t2_map = t2_map_nib.get_fdata()

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
                gm_mask = gm_mask_nib.get_fdata() > 0.8
                atlas_roi_thres = 0.3
                r2_thres = 150

                # Extract T2 values for each cortical region
                region_dict = dict()
                region_dict_left = dict()
                region_dict_right = dict()

                region_subject_mean_dict = dict()
                region_subject_mean_dict_left = dict()
                region_subject_mean_dict_right = dict()

                for cortical_region in cortical_regions:
                    roi_probseg = cortical_probseg[:, :, :,
                                  cortical_region["label"]]
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(np.bool)
                    t2_map_roi = t2_map[roi_mask]
                    t2_map_roi = t2_map_roi[t2_map_roi < r2_thres]
                    region_dict[cortical_region["name"]] = t2_map_roi
                    region_subject_mean_dict[cortical_region["name"]] = np.mean(
                        t2_map_roi)

                    roi_probseg = cortical_left_probseg[:, :, :,
                                  cortical_region["label"]]
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(np.bool)
                    t2_map_roi = t2_map[roi_mask]
                    t2_map_roi = t2_map_roi[t2_map_roi < r2_thres]
                    region_dict_left[cortical_region["name"]] = t2_map_roi
                    region_subject_mean_dict_left[
                        cortical_region["name"]] = np.mean(
                        t2_map_roi)

                    roi_probseg = cortical_right_probseg[:, :, :,
                                  cortical_region["label"]]
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(np.bool)
                    t2_map_roi = t2_map[roi_mask]
                    t2_map_roi = t2_map_roi[t2_map_roi < r2_thres]
                    region_dict_right[cortical_region["name"]] = t2_map_roi
                    region_subject_mean_dict_right[
                        cortical_region["name"]] = np.mean(
                        t2_map_roi)

                for subcortical_region in subcortical_regions:
                    labels = subcortical_region["label"]

                    left_label = labels[0]
                    roi_probseg = subcortical_probseg[:, :, :,
                                  left_label]
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(np.bool)
                    t2_map_roi = t2_map[roi_mask]
                    t2_map_roi = t2_map_roi[t2_map_roi < r2_thres]
                    t2_map_roi = t2_map_roi[~np.isnan(t2_map_roi)]
                    region_dict_left[subcortical_region["name"]] = t2_map_roi
                    region_subject_mean_dict_left[
                        subcortical_region["name"]] = np.mean(
                        t2_map_roi)

                    right_label = labels[1]
                    roi_probseg = subcortical_probseg[:, :, :,
                                  right_label]
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(np.bool)
                    t2_map_roi = t2_map[roi_mask]
                    t2_map_roi = t2_map_roi[t2_map_roi < r2_thres]
                    t2_map_roi = t2_map_roi[~np.isnan(t2_map_roi)]
                    region_dict_right[subcortical_region["name"]] = t2_map_roi
                    region_subject_mean_dict_right[
                        subcortical_region["name"]] = np.mean(
                        t2_map_roi)

                    roi_probseg = np.zeros(subcortical_probseg.shape[:3])
                    for label in labels:
                        roi_probseg += subcortical_probseg[:, :, :,
                                       label]
                        roi_mask = roi_probseg / 100
                        roi_mask[roi_mask < atlas_roi_thres] = 0
                        roi_mask = np.logical_and(roi_mask, gm_mask)
                        roi_mask = roi_mask.astype(np.bool)
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(np.bool)
                    t2_map_roi = t2_map[roi_mask]
                    t2_map_roi = t2_map_roi[t2_map_roi < r2_thres]
                    t2_map_roi = t2_map_roi[~np.isnan(t2_map_roi)]
                    region_dict[subcortical_region["name"]] = t2_map_roi
                    region_subject_mean_dict[
                        subcortical_region["name"]] = np.mean(
                        t2_map_roi)

                # Save region data
                region_data[site_index, subject_run_index] = region_dict
                region_data_left[
                    site_index, subject_run_index] = region_dict_left
                region_data_right[
                    site_index, subject_run_index] = region_dict_right
                region_subject_mean_data[
                    site_index, subject_run_index] = region_subject_mean_dict
                region_subject_mean_data_left[
                    site_index, subject_run_index] = region_subject_mean_dict_left
                region_subject_mean_data_right[
                    site_index, subject_run_index] = region_subject_mean_dict_right
                print("    - T2 values extracted and saved for regions")

        print("\nData collection completed. Now pooling data across sites.")

        # Pool T2 values over all subjects for each site and region
        region_names = [region["name"] for region in cortical_regions] + [
            region["name"] for region in subcortical_regions]
        pooled_region_data = {region: {site: [] for site in dataset_names} for
                              region in region_names}
        pooled_region_data_left = {region: {site: [] for site in dataset_names}
                                   for
                                   region in region_names}
        pooled_region_data_right = {region: {site: [] for site in dataset_names}
                                    for
                                    region in region_names}

        pooled_region_data_subject_means = {
            region: {site: [] for site in dataset_names} for
            region in region_names}
        pooled_region_data_subject_means_left = {
            region: {site: [] for site in dataset_names} for
            region in region_names}
        pooled_region_data_subject_means_right = {
            region: {site: [] for site in dataset_names} for
            region in region_names}

        for site_index, site_name in enumerate(dataset_names):
            for subject_run_index in range(len(subject_run_combinations)):
                region_dict = region_data[site_index, subject_run_index]
                region_dict_left = region_data_left[
                    site_index, subject_run_index]
                region_dict_right = region_data_right[
                    site_index, subject_run_index]
                region_dict_subject_means = region_subject_mean_data[
                    site_index, subject_run_index]
                region_dict_subject_means_left = region_subject_mean_data_left[
                    site_index, subject_run_index]
                region_dict_subject_means_right = \
                region_subject_mean_data_right[
                    site_index, subject_run_index]

                if region_dict is not None:
                    for region_name in region_names:
                        pooled_region_data[region_name][site_name].extend(
                            region_dict[region_name])
                        pooled_region_data_subject_means[region_name][
                            site_name].append(
                            region_dict_subject_means[region_name])
                if region_dict_left is not None:
                    for region_name in region_names:
                        pooled_region_data_left[region_name][site_name].extend(
                            region_dict_left[region_name])
                        pooled_region_data_subject_means_left[region_name][
                            site_name].append(
                            region_dict_subject_means_left[region_name])
                if region_dict_right is not None:
                    for region_name in region_names:
                        pooled_region_data_right[region_name][
                            site_name].extend(
                            region_dict_right[region_name])
                        pooled_region_data_subject_means_right[region_name][
                            site_name].append(
                            region_dict_subject_means_right[region_name])

        print("Data pooling completed. Now generating boxplots.")

        # Save the results to a pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                "region_data": region_data,
                "region_data_left": region_data_left,
                "region_data_right": region_data_right,
                "region_subject_mean_data": region_subject_mean_data,
                "region_subject_mean_data_left": region_subject_mean_data_left,
                "region_subject_mean_data_right": region_subject_mean_data_right,
                "pooled_region_data": pooled_region_data,
                "pooled_region_data_left": pooled_region_data_left,
                "pooled_region_data_right": pooled_region_data_right,
                "pooled_region_data_subject_means": pooled_region_data_subject_means,
                "pooled_region_data_subject_means_left": pooled_region_data_subject_means_left,
                "pooled_region_data_subject_means_right": pooled_region_data_subject_means_right,
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
        region_subject_mean_data = data["region_subject_mean_data"]
        region_subject_mean_data_left = data["region_subject_mean_data_left"]
        region_subject_mean_data_right = data["region_subject_mean_data_right"]
        pooled_region_data = data["pooled_region_data"]
        pooled_region_data_left = data["pooled_region_data_left"]
        pooled_region_data_right = data["pooled_region_data_right"]
        pooled_region_data_subject_means = data[
            "pooled_region_data_subject_means"]
        pooled_region_data_subject_means_left = data[
            "pooled_region_data_subject_means_left"]
        pooled_region_data_subject_means_right = data[
            "pooled_region_data_subject_means_right"]
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

    # Create a 2xN grid where the first row is for boxplots and the second row is for inter-hemispheric plots
    fig, axes = plt.subplots(2, num_sites, figsize=(15, 10), sharey='row')

    # First row: Boxplots for each site
    for site_idx, site_name in enumerate(dataset_names):
        region_values = pooled_region_data

        # Prepare data for the boxplot (grouped by region)
        data = [pooled_region_data[region_name][site_name] for region_name in
                region_names]

        flierprops = dict(marker='.', markerfacecolor='black', markersize=3,
                          linestyle='none')

        # Create a boxplot for this site
        bplot = axes[0, site_idx].boxplot(data, labels=region_names,
                                          flierprops=flierprops,
                                          patch_artist=True)

        # Color each boxplot according to the region color
        for patch, color in zip(bplot['boxes'], region_colors):
            patch.set_facecolor(color)

        # Compute and overlay the per-subject means on top of the boxplot
        for i, region_name in enumerate(region_names):
            per_subject_means = pooled_region_data_subject_means[region_name][
                site_name]

            # Overlay the per-subject mean values using scatter plot
            positions = np.full(len(per_subject_means),
                                i + 1)  # x-axis position for each region

            subject_markers = ['o', '^', 'v', 's']
            for subject_index, subject_mean in enumerate(per_subject_means):
                axes[0, site_idx].scatter(positions[subject_index], subject_mean, alpha=0.8,
                                          color='white', s=30, edgecolor='black',
                                          zorder=2, marker=subject_markers[subject_index])

        # Add titles and labels
        axes[0, site_idx].set_title(f"Site: {site_name}")
        axes[0, site_idx].set_xticklabels(region_names, rotation=45, ha='right')
        axes[0, site_idx].set_ylim([0, 40])  # Adjust the y-axis limit as needed
        axes[0, site_idx].set_xlabel("Region")
        if site_idx == 0:
            axes[0, site_idx].set_ylabel("R2 [1/s]")

    # Second row: Inter-hemispheric differences for each site
    for site_idx, site_name in enumerate(dataset_names):
        # left_means = np.array(
        #     [np.mean(pooled_region_data_left[region_name][site_name]) for
        #      region_name in region_names])
        # right_means = np.array(
        #     [np.mean(pooled_region_data_right[region_name][site_name]) for
        #      region_name in region_names])

        hemisphere_differences_per_region = []
        hemisphere_differences_per_region_mean = []
        hemisphere_differences_per_region_std = []

        for region_name in region_names:
            left_means = np.array(
                pooled_region_data_subject_means_left[region_name][site_name])
            right_means = np.array(
                pooled_region_data_subject_means_right[region_name][site_name])
            hemisphere_differences_roi = left_means - right_means
            hemisphere_differences_per_region.append(hemisphere_differences_roi)
            hemisphere_differences_per_region_mean.append(
                np.mean(hemisphere_differences_roi))
            hemisphere_differences_per_region_std.append(
                np.std(hemisphere_differences_roi))

        # left_means = np.array(
        #     [np.mean(pooled_region_data_left[region_name][site_name]) for
        #      region_name in region_names])
        # right_means = np.array(
        #     [np.mean(pooled_region_data_right[region_name][site_name]) for
        #      region_name in region_names])

        # Compute the differences between left and right hemispheres
        # hemisphere_differences = left_means - right_means

        # Plot the differences as horizontal bars with the same color as the region
        for i, diff in enumerate(hemisphere_differences_per_region_mean):
            axes[1, site_idx].barh(i,
                                   hemisphere_differences_per_region_mean[i],
                                   color=region_colors[i],
                                   edgecolor='black')

        # Invert y-axis to match the order of regions with the boxplots
        axes[1, site_idx].invert_yaxis()

        # Add labels and grid
        axes[1, site_idx].set_yticks(range(len(region_names)))
        axes[1, site_idx].set_yticklabels(region_names)
        axes[1, site_idx].set_xlim(
            [-2.5, 2.5])  # Adjust based on expected difference range
        axes[1, site_idx].axvline(0, color='black',
                                  linewidth=0.5)  # Vertical line at zero
        axes[1, site_idx].set_title(f"Site: {site_name}")
        axes[1, site_idx].set_xlabel("Interhemispheric (left-right)\n R2 difference [1/s]")
        if site_idx == 0:
            axes[1, site_idx].set_ylabel("Region")

    # Adjust the layout
    plt.tight_layout()

    # Save or show the plot
    plt.savefig(os.path.join(out_dir,
                             "t2_region_comparison_boxplot_and_interhemispheric_per_site.png"),
                dpi=300)
    plt.show()

    print(
        "Boxplots and inter-hemispheric difference plots generated and saved.")
