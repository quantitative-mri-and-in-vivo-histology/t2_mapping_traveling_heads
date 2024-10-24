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
        "/media/laurin/Elements/Travel_Head_Study/clean/results/Hamburg_Prisma_3T_dzne",
        "/media/laurin/Elements/Travel_Head_Study/clean/results/London_Kings_Vida_3T",
        "/media/laurin/Elements/Travel_Head_Study/clean/results/Hamburg_Prisma_3T_ssfp"
    ]

    dataset_names = [
        "3D-EPI (Reference)",
        "3D-EPI (Prisma)",
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
                                                       "registeredMidScanRescan")
            subject_registered_data_dirs.append(subject_registered_data_dir)

            print(f"Loading BIDS layout for dataset: {dataset_dir}")
            mni_result_layouts.append(
                BIDSLayout(subject_registered_data_dir,
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
                    run=None,
                    desc="cortical",
                    suffix="probseg",
                    extension="nii.gz")
                assert (len(cortical_probseg_files) == 1)
                cortical_probseg_file = cortical_probseg_files[0]

                subcortical_probseg_files = layout.get(
                    **subject_dict,
                    run=None,
                    desc="subcortical",
                    suffix="probseg",
                    extension="nii.gz")
                assert (len(subcortical_probseg_files) == 1)
                subcortical_probseg_file = subcortical_probseg_files[0]

                gm_probseg_files = layout.get(
                    **subject_dict,
                    run=None,
                    desc="gmPosterior",
                    suffix="probseg",
                    extension="nii.gz")
                assert (len(gm_probseg_files) == 1)
                gm_probseg_file = gm_probseg_files[0]

                r2_scan_map_files = layout.get(
                    **subject_dict,
                    run=1,
                    suffix="R2map",
                    extension="nii.gz")
                assert (len(r2_scan_map_files) == 1)
                r2_scan_map_file = r2_scan_map_files[0]

                r2_rescan_map_files = layout.get(
                    **subject_dict,
                    run=2,
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
                gm_mask = gm_mask_nib.get_fdata() > 0.8
                atlas_roi_thres = 0.3
                r2_thres = 150

                # Extract T2 values for each cortical region
                region_dict = dict()

                for cortical_region in cortical_regions:
                    roi_probseg = cortical_probseg[:, :, :,
                                  cortical_region["label"]]
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(np.bool)

                    r2_map_scan_roi = r2_map_scan[roi_mask]
                    # r2_map_scan_roi = r2_map_scan_roi[r2_map_scan_roi < r2_thres]

                    r2_map_rescan_roi = r2_map_rescan[roi_mask]
                    # r2_map_rescan_roi = r2_map_rescan_roi[r2_map_rescan_roi < r2_thres]

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
                        roi_mask = roi_mask.astype(np.bool)
                    roi_mask = roi_probseg / 100
                    roi_mask[roi_mask < atlas_roi_thres] = 0
                    roi_mask = np.logical_and(roi_mask, gm_mask)
                    roi_mask = roi_mask.astype(np.bool)

                    r2_map_scan_roi = r2_map_scan[roi_mask]
                    # r2_map_scan_roi = r2_map_scan_roi[
                        # r2_map_scan_roi < r2_thres]

                    r2_map_rescan_roi = r2_map_rescan[roi_mask]
                    # r2_map_rescan_roi = r2_map_rescan_roi[
                    #     r2_map_rescan_roi < r2_thres]
                    region_dict[subcortical_region["name"]] = [r2_map_scan_roi, r2_map_rescan_roi]

                # Save region data
                region_data[site_index, subject_index] = region_dict
                print("    - R2 values extracted and saved for regions")

        # Pool T2 values over all subjects for each site and region
        region_names = [region["name"] for region in cortical_regions] + [
            region["name"] for region in subcortical_regions]
        pooled_region_data = {region: {site: [] for site in dataset_names} for
                              region in region_names}

        for site_index, site_name in enumerate(dataset_names):
            for subject_run_index in range(len(subject_dicts)):
                region_dict = region_data[site_index, subject_run_index]
                if region_dict is not None:
                    for region_name in region_names:
                        pooled_region_data[region_name][site_name].extend(
                            region_dict[region_name])

            # Now proceed to create Bland-Altman plots
        fig, axs = plt.subplots(len(dataset_names), len(region_names), figsize=(20, 20))

        for site_index, site_name in enumerate(dataset_names):
            for region_index, region_name in enumerate(region_names):
                ax = axs[site_index, region_index]
                r2_scan_roi, r2_rescan_roi = region_data[site_index, 0][
                    region_name]  # Assuming subject index 0

                r2_scan_roi = np.array(r2_scan_roi)
                r2_rescan_roi = np.array(r2_rescan_roi)

                valid_idx = np.logical_and(r2_scan_roi < r2_thres, r2_rescan_roi < r2_thres)
                r2_scan_roi = r2_scan_roi[valid_idx]
                r2_rescan_roi = r2_rescan_roi[valid_idx]

                valid_idx = np.logical_and(np.isfinite(r2_scan_roi), np.isfinite(r2_rescan_roi))
                r2_scan_roi = r2_scan_roi[valid_idx]
                r2_rescan_roi = r2_rescan_roi[valid_idx]

                r2_median = np.median(np.ravel([r2_scan_roi, r2_rescan_roi]))
                r2_scan_roi = r2_scan_roi/r2_median*100
                r2_rescan_roi = r2_rescan_roi/r2_median*100

                mean = np.mean([r2_scan_roi, r2_rescan_roi], axis=0)
                diff = r2_scan_roi - r2_rescan_roi  # Difference between data1 and data2
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)

                # Bland-Altman plotssc
                ax.scatter(mean, diff, s=10, alpha=0.5)
                ax.axhline(mean_diff, color='gray', linestyle='--')
                ax.axhline(mean_diff + 1.96 * std_diff, color='red',
                           linestyle='--')
                ax.axhline(mean_diff - 1.96 * std_diff, color='red',
                           linestyle='--')

                ax.set_xlim([0, 200])
                ax.set_ylim([-125, 125])

                # ax.set_xlim([0, 50])
                # ax.set_ylim([-0.25, 0.25])

                if region_index == 0:  # Add site name to the first column (leftmost)
                    ax.set_ylabel(site_name, fontsize=16,
                                  labelpad=10, rotation=90, verticalalignment='center')
                else:
                    ax.set_xlabel('Mean of Scan and Rescan R2 in percent\nnormalized to ROi median')
                    ax.set_ylabel('Difference (Scan - Rescan) in percent\nnormalized to ROI median')


                if site_index == 0:  # Add site name to the first column (leftmost)
                    ax.set_title(f"{region_name}", fontsize=16)

                    # ax.set_ylabel(site_name, fontsize=16,
                    #               labelpad=10, rotate=90,
                    #               verticalalignment='center')




        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(out_dir, "bland_altman_plots_normalized.png"))
        print(
            f"Bland-Altman plots saved to {os.path.join(out_dir, 'bland_altman_plots.png')}")

        # Show the figure if desired
        plt.show()
