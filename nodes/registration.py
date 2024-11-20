from nipype.interfaces.ants import Registration as Registration


def create_default_ants_rigid_registration_node():
    ants_reg_params = dict(
        dimension=3,  # 3D images
        output_transform_prefix='output_prefix_',  # Prefix for outputs
        transforms=['Rigid'],  # Rigid transformation
        transform_parameters=[(0.1,)],  # Step size for rigid transformation
        metric=['MI'],  # Mutual Information
        metric_weight=[1],  # Weight of the metric
        radius_or_number_of_bins=[32],  # Number of histogram bins for MI
        sampling_strategy=['Regular'],  # Regular sampling
        sampling_percentage=[0.25],  # Sampling percentage
        convergence_threshold=[1e-6],  # Convergence threshold
        convergence_window_size=[10],  # Convergence window size
        number_of_iterations=[[1000, 500, 250, 100]],
        shrink_factors=[[8, 4, 2, 1]],  # Shrink factors for multi-resolution
        smoothing_sigmas=[[3, 2, 1, 0]],  # Smoothing sigmas for multi-resolution
        interpolation='Linear',  # Bspline interpolation
        collapse_output_transforms=True,
        output_warped_image='output_warped_image.nii.gz'
    )
    ants_registration = Registration(**ants_reg_params)
    return ants_registration


def create_default_ants_rigid_affine_registration_node():
    ants_reg_params = dict(
        dimension=3,  # 3D images
        output_transform_prefix='output_prefix_',  # Prefix for outputs
        transforms=['Rigid', 'Affine'],  # Rigid transformation
        transform_parameters=[(0.1,), (0.1,)],  # Step size for rigid transformation
        metric=['MI', 'MI'],  # Mutual Information
        metric_weight=[1, 1],  # Weight of the metric
        radius_or_number_of_bins=[32, 32],  # Number of histogram bins for MI
        sampling_strategy=['Regular', 'Regular'],  # Regular sampling
        sampling_percentage=[0.25, 0.25],  # Sampling percentage
        convergence_threshold=[1e-6, 1e-6],  # Convergence threshold
        convergence_window_size=[10, 10],  # Convergence window size
        number_of_iterations=[[1000, 500, 250, 100], [1000, 500, 250, 100]],
        shrink_factors=[[8, 4, 2, 1], [8, 4, 2, 1]],  # Shrink factors for multi-resolution
        smoothing_sigmas=[[3, 2, 1, 0], [3, 2, 1, 0]],  # Smoothing sigmas for multi-resolution
        interpolation='Linear',  # Bspline interpolation
        collapse_output_transforms=True,
        output_warped_image='output_warped_image.nii.gz'
    )
    ants_registration = Registration(**ants_reg_params)
    return ants_registration


def create_default_ants_rigid_affine_syn_registration_node():

    ants_reg_params = dict(
        dimension=3,  # 3D registration
        output_transform_prefix='output_prefix_',  # Prefix for output files
        transforms=['Rigid', 'Affine', 'SyN'],  # Transformation types
        transform_parameters=[(0.1,), (0.1,), (0.1, 3, 0)],
        # Parameters for each transform
        metric=['MI', 'MI', 'CC'],
        # Metrics for each stage: MI for Rigid/Affine, CC for SyN
        metric_weight=[1, 1, 1],  # Weights for the metrics
        radius_or_number_of_bins=[32, 32, 4],
        # Number of bins for MI and radius for CC
        sampling_strategy=['Regular', 'Regular', None],
        # Sampling strategies for each stage
        sampling_percentage=[0.25, 0.25, None],  # Sampling percentages for MI
        convergence_threshold=[1e-6, 1e-6, 1e-6],  # Convergence thresholds
        convergence_window_size=[10, 10, 10],  # Convergence window sizes
        number_of_iterations=[[1000, 500, 250, 100], [1000, 500, 250, 100],
                              [120, 90, 60, 30]],
        # Iterations for each resolution level
        shrink_factors=[[8, 4, 2, 1], [8, 4, 2, 1], [8, 4, 2, 1]],
        # Shrink factors for the multi-resolution scheme
        smoothing_sigmas=[[3, 2, 1, 0], [3, 2, 1, 0], [3, 2, 1, 0]],
        winsorize_upper_quantile=0.995,
        winsorize_lower_quantile=0.005,
        # Smoothing sigmas for the multi-resolution scheme
        interpolation='Linear',  # Linear interpolation
        output_warped_image='output_warped_image.nii.gz',  # Output warped image
        output_inverse_warped_image='output_inverse_warped_image.nii.gz',
        collapse_output_transforms=True,
    )
    ants_registration = Registration(**ants_reg_params)
    return ants_registration

def create_default_ants_rigid_double_affine_syn_registration_node():

    ants_reg_params = dict(
        dimension=3,  # 3D registration
        output_transform_prefix='output_prefix_',  # Prefix for output files
        transforms=['Rigid', 'Affine', 'Affine', 'SyN'],  # Transformation types
        transform_parameters=[(0.1,), (0.1,), (0.1,), (0.1, 3, 0)],
        # Parameters for each transform
        metric=['MI', 'MI', 'MI', 'CC'],
        # Metrics for each stage: MI for Rigid/Affine, CC for SyN
        metric_weight=[1, 1, 1, 1], # Weights for the metrics
        radius_or_number_of_bins=[32, 32, 32, 4],
        # Number of bins for MI and radius for CC
        sampling_strategy=['Regular', 'Regular', 'Regular', None],
        # Sampling strategies for each stage
        sampling_percentage=[1.0,1.0,1.0, None],  # Sampling percentages for MI
        convergence_threshold=[1e-6, 1e-6, 1e-6, 1e-6],  # Convergence thresholds
        convergence_window_size=[10, 10, 10, 10],  # Convergence window sizes
        number_of_iterations=[[1000, 500, 250, 100], [1000, 500, 250, 100],
                              [1000, 500, 250, 100], [100, 70, 50, 20]],
        # Iterations for each resolution level
        shrink_factors=[[8, 4, 2, 1], [8, 4, 2, 1], [8, 4, 2, 1], [6, 4, 2, 1]],
        # Shrink factors for the multi-resolution scheme
        smoothing_sigmas=[[3, 2, 1, 0], [3, 2, 1, 0], [3, 2, 1, 0], [3, 2, 1, 0]],
        # Smoothing sigmas for the multi-resolution scheme
        interpolation='Linear',  # Linear interpolation
        output_warped_image='output_warped_image.nii.gz',  # Output warped image
        output_inverse_warped_image='output_inverse_warped_image.nii.gz',
        collapse_output_transforms = True,
    )
    ants_registration = Registration(**ants_reg_params)
    return ants_registration

