Node: register_t1w (ants)
=========================


 Hierarchy : register_maps_to_mni.register_t1w
 Exec ID : register_t1w.a0


Original Inputs
---------------


* args : <undefined>
* collapse_output_transforms : True
* convergence_threshold : [1e-06, 1e-06, 1e-06]
* convergence_window_size : [10, 10, 10]
* dimension : 3
* environ : {'NSLOTS': '1'}
* fixed_image : ['/home/laurin/fsl/data/standard/MNI152_T1_1mm.nii.gz']
* fixed_image_mask : <undefined>
* fixed_image_masks : <undefined>
* float : <undefined>
* initial_moving_transform : <undefined>
* initial_moving_transform_com : True
* initialize_transforms_per_stage : False
* interpolation : Linear
* interpolation_parameters : <undefined>
* invert_initial_moving_transform : <undefined>
* metric : ['MI', 'MI', 'CC']
* metric_item_trait : <undefined>
* metric_stage_trait : <undefined>
* metric_weight : [1.0, 1.0, 1.0]
* metric_weight_item_trait : 1.0
* metric_weight_stage_trait : <undefined>
* moving_image : ['/media/laurin/Elements/Travel_Head_Study/clean/results/Hamburg_Prisma_3T_dzne/processed/sub-phy004/ses-001/anat/sub-phy004_ses-001_acq-T1wRef_desc-preproc_T1w.nii.gz']
* moving_image_mask : <undefined>
* moving_image_masks : <undefined>
* num_threads : 1
* number_of_iterations : [[1000, 500, 250, 100], [1000, 500, 250, 100], [100, 70, 50, 20]]
* output_inverse_warped_image : output_inverse_warped_image.nii.gz
* output_transform_prefix : output_prefix_
* output_warped_image : output_warped_image.nii.gz
* radius_bins_item_trait : 5
* radius_bins_stage_trait : <undefined>
* radius_or_number_of_bins : [32, 32, 4]
* random_seed : <undefined>
* restore_state : <undefined>
* restrict_deformation : <undefined>
* sampling_percentage : [0.25, 0.25, None]
* sampling_percentage_item_trait : <undefined>
* sampling_percentage_stage_trait : <undefined>
* sampling_strategy : ['Regular', 'Regular', None]
* sampling_strategy_item_trait : <undefined>
* sampling_strategy_stage_trait : <undefined>
* save_state : <undefined>
* shrink_factors : [[8, 4, 2, 1], [8, 4, 2, 1], [6, 4, 2, 1]]
* sigma_units : <undefined>
* smoothing_sigmas : [[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]]
* transform_parameters : [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
* transforms : ['Rigid', 'Affine', 'BSplineSyN']
* use_estimate_learning_rate_once : <undefined>
* use_histogram_matching : True
* verbose : False
* winsorize_lower_quantile : 0.005
* winsorize_upper_quantile : 0.995
* write_composite_transform : False

