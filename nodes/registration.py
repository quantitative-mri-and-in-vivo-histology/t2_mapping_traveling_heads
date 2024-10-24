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
    reg_node = Registration(name=name)
    reg_node.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg_node.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3, 0)]
    reg_node.inputs.number_of_iterations = [[1000, 500, 250, 100],
                                            [1000, 500, 250, 100],
                                            [200, 200, 200]]
    reg_node.inputs.dimension = 3
    reg_node.inputs.write_composite_transform = True
    reg_node.inputs.collapse_output_transforms = True
    reg_node.inputs.initialize_transforms_per_stage = False
    reg_node.inputs.metric = ['Mattes', 'Mattes', 'CC']
    reg_node.inputs.metric_weight = [1, 1, 1]
    reg_node.inputs.radius_or_number_of_bins = [32, 32, 4]
    reg_node.inputs.sampling_strategy = ['Regular', 'Regular', 'None']
    reg_node.inputs.sampling_percentage = [0.25, 0.25, None]
    reg_node.inputs.convergence_threshold = [1e-6, 1e-6, 1e-6]
    reg_node.inputs.convergence_window_size = [10, 10, 10]
    reg_node.inputs.smoothing_sigmas = [[3, 2, 1, 0], [3, 2, 1, 0], [2, 1, 0]]
    reg_node.inputs.sigma_units = ['vox', 'vox', 'vox']
    reg_node.inputs.shrink_factors = [[8, 4, 2, 1], [8, 4, 2, 1], [4, 2, 1]]
    reg_node.inputs.use_estimate_learning_rate_once = [True, True, False]
    reg_node.inputs.use_histogram_matching = [False, False, True]
    reg_node.inputs.output_warped_image = True
    return reg_node


def create_default_ants_rigid_affine_bsplinesyn_registration_node():
    reg_node = Registration(name=name)
    reg_node.inputs.transforms = ['Rigid', 'Affine', 'BSplineSyN']
    reg_node.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3, 0)]
    reg_node.inputs.number_of_iterations = [[1000, 500, 250, 100],
                                            [1000, 500, 250, 100],
                                            [200, 200, 200]]
    reg_node.inputs.dimension = 3
    reg_node.inputs.write_composite_transform = True
    reg_node.inputs.collapse_output_transforms = True
    reg_node.inputs.initialize_transforms_per_stage = False
    reg_node.inputs.metric = ['Mattes', 'Mattes', 'CC']
    reg_node.inputs.metric_weight = [1, 1, 1]
    reg_node.inputs.radius_or_number_of_bins = [32, 32, 4]
    reg_node.inputs.sampling_strategy = ['Regular', 'Regular', 'None']
    reg_node.inputs.sampling_percentage = [0.25, 0.25, None]
    reg_node.inputs.convergence_threshold = [1e-6, 1e-6, 1e-6]
    reg_node.inputs.convergence_window_size = [10, 10, 10]
    reg_node.inputs.smoothing_sigmas = [[3, 2, 1, 0], [3, 2, 1, 0], [2, 1, 0]]
    reg_node.inputs.sigma_units = ['vox', 'vox', 'vox']
    reg_node.inputs.shrink_factors = [[8, 4, 2, 1], [8, 4, 2, 1], [4, 2, 1]]
    reg_node.inputs.use_estimate_learning_rate_once = [True, True, False]
    reg_node.inputs.use_histogram_matching = [False, False, True]
    reg_node.inputs.output_warped_image = True
    return reg_node
