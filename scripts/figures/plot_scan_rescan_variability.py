import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from bids.layout import BIDSLayout
# Create custom legend handles
from matplotlib.patches import Patch

if __name__ == "__main__":

    out_dir = "../../data/figures/quantitative"
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

    registered_data_dirs = []
    mni_result_layouts = []
    for dataset_dir in results_dataset_dirs:
        registered_data_dir = os.path.join(dataset_dir,
                                           "registered")
        registered_data_dirs.append(registered_data_dir)

        print(f"Loading BIDS layout for dataset: {dataset_dir}")
        mni_result_layouts.append(
            BIDSLayout(registered_data_dir,
                       validate=False))

    sub_ses_run_dicts = []
    subjects = mni_result_layouts[0].get_subjects()

    subject_dicts = [
        dict(subject="phy001"),
    ]

    region_data = np.empty(
        (len(mni_result_layouts), len(subject_dicts)),
        dtype=object)

    # Start processing each site
    for site_index, layout in enumerate(mni_result_layouts):
        print(
            f"\nProcessing site {site_index + 1}/{len(mni_result_layouts)}: {dataset_names[site_index]}")

        for subject_index, subject_dict in enumerate(
                subject_dicts):

            # Load probability segmentation and T2 map files
            cortical_probseg_files = layout.get(
                **subject_dict,
                run=1,
                space="midspaceRuns",
                label="cortical",
                suffix="probseg",
                extension="nii.gz")
            assert (len(cortical_probseg_files) == 1)
            cortical_probseg_file = cortical_probseg_files[0]

            subcortical_probseg_files = layout.get(
                **subject_dict,
                run=1,
                space="midspaceRuns",
                label="subcortical",
                suffix="probseg",
                extension="nii.gz")
            assert (len(subcortical_probseg_files) == 1)
            subcortical_probseg_file = subcortical_probseg_files[0]

            gm_probseg_files = layout.get(
                **subject_dict,
                run=None,
                space="midspaceRuns",
                label="gm",
                suffix="probseg",
                extension="nii.gz")
            assert (len(gm_probseg_files) == 1)
            gm_probseg_file = gm_probseg_files[0]

            r2_scan_map_files = layout.get(
                **subject_dict,
                run=1,
                space="midspaceRuns",
                suffix="R2map",
                extension="nii.gz")
            assert (len(r2_scan_map_files) == 1)
            r2_scan_map_file = r2_scan_map_files[0]

            r2_rescan_map_files = layout.get(
                **subject_dict,
                run=2,
                space="midspaceRuns",
                suffix="R2map",
                extension="nii.gz")
            assert (len(r2_rescan_map_files) == 1)
            r2_rescan_map_file = r2_rescan_map_files[0]

            print("    - Loading R2 map and probability segmentation files")

            # Load the files into memory
            r2_map_scan_nib = nib.load(r2_scan_map_file)
            r2_map_scan = r2_map_scan_nib.get_fdata()

            r2_map_rescan_nib = nib.load(r2_rescan_map_file)
            r2_map_rescan = r2_map_rescan_nib.get_fdata()

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

                r2_map_scan_roi = r2_map_scan[roi_mask]
                r2_map_rescan_roi = r2_map_rescan[roi_mask]
                region_dict[cortical_region["name"]] = [r2_map_scan_roi,
                                                        r2_map_rescan_roi]

            for subcortical_region in subcortical_regions:
                labels = subcortical_region["label"]
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

                r2_map_scan_roi = r2_map_scan[roi_mask]
                r2_map_rescan_roi = r2_map_rescan[roi_mask]
                region_dict[subcortical_region["name"]] = [r2_map_scan_roi,
                                                           r2_map_rescan_roi]

            # Save region data
            region_data[site_index, subject_index] = region_dict
            print("    - R2 values extracted and saved for regions")

    # Pool T2 values over all subjects for each site and region
    region_names = [region["name"] for region in cortical_regions] + [
        region["name"] for region in subcortical_regions]


    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))

    pooled_region_data = {region: {site: [] for site in dataset_names} for
                          region in region_names}
    for site_index, site_name in enumerate(dataset_names):
        for subject_run_index in range(len(subject_dicts)):
            region_dict = region_data[site_index, subject_run_index]
            if region_dict is not None:
                for region_name in region_names:
                    pooled_region_data[region_name][site_name].extend(
                        region_dict[region_name])

    vals_per_site_region_mean = np.zeros(
        (region_data.shape[0], len(region_names)),
        dtype=float)
    vals_per_site_region_std= np.zeros(
        (region_data.shape[0], len(region_names)),
        dtype=float)
    for site_idx, site_name in enumerate(dataset_names):
        for region_index, region_name in enumerate(region_names):
            vals_scan, vals_rescan = region_data[site_idx, 0][region_name]

            vals_scan = np.array(vals_scan)
            vals_rescan = np.array(vals_rescan)

            valid_idx = np.logical_and(vals_scan < r2_thres,
                                       vals_rescan < r2_thres)
            vals_scan = vals_scan[valid_idx]
            vals_rescan = vals_rescan[valid_idx]

            valid_idx = np.logical_and(np.isfinite(vals_scan),
                                       np.isfinite(vals_rescan))
            vals_scan = vals_scan[valid_idx]
            vals_rescan = vals_rescan[valid_idx]

            diffs = vals_scan - vals_rescan
            all_vals = [vals_scan, vals_rescan]
            vals_per_site_region_mean[site_idx, region_index] = np.median(all_vals)
            vals_per_site_region_std[site_idx, region_index] = (np.percentile(all_vals, 75) - np.percentile(all_vals, 25))/np.sqrt(2)

    intra_region_var_scan_rescan = vals_per_site_region_std/vals_per_site_region_mean

    bland_altman_ax = ax
    bland_altman_ax.bar(range(0, len(dataset_names)),
                            100 * np.mean(intra_region_var_scan_rescan, 1),
                            yerr=100 * np.std(intra_region_var_scan_rescan, 1),
                            edgecolor='black', color=colors)
    bland_altman_ax.set_xlabel("Site")
    bland_altman_ax.set_ylabel("Scan-rescan CoV [%]")
    bland_altman_ax.set_xticks([])
    bland_altman_ax.set_xticklabels([])
    bland_altman_ax.set_ylim([0, 35])

    dataset_names = [
        "MPR (reference site)",
        "MPR (target site)",
        "CSMT-JSR (reference site)",
        "JSR (target site)"
    ]

    legend_elements = []
    for dataset_name, color in zip(dataset_names, colors):
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=dataset_name))

    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.03), frameon=False)

    # Save or show the plot
    plt.tight_layout(
        rect=[0.01, 0, 1, 0.92], pad=0.1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(os.path.join(out_dir, "r2_variability_scan_rescan.png"),
                dpi=300)
