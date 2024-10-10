import os
from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
from nipype import Node, Function
from datasets.three_dim_epi_dataset import ThreeDimEpiDataset
from utils.processing import unwrap_phase_b0_siemens, cut_and_merge_image, \
    create_brain_mask_from_anatomical_b1, inpaint
from workflows.fieldmap_workflows import correct_b1_with_b0


def simple_copy(in_file, target):
    """
    Copies a file to the specified target location, creating directories as needed.

    Parameters:
        in_file (str): Path to the input file that needs to be copied.
        target (str): The full path (including filename) where the file should be copied.

    Returns:
        str: The path to the copied file (i.e., the target).
    """
    import os
    import shutil

    # Ensure the directory for the target path exists.
    os.makedirs(os.path.dirname(target), exist_ok=True)

    # Copy the file to the target path.
    shutil.copy(in_file, target)

    print(target)

    # Return the target path.
    return target


class DzneThreeDimEpiDataset(ThreeDimEpiDataset):

    def __init__(self, bids_root, derivatives_output_folder, derivatives=None):
        # Call the constructor of the parent class (3dEpiBids)
        super().__init__(bids_root=bids_root,
                         derivatives_output_folder=derivatives_output_folder,
                         derivatives=derivatives)

    def get_b0_map_siemens(self, subject, session, run=None,
                           allow_multiple_files=False):
        return self.get_file(
            subject=subject,
            session=session,
            suffix='T2w',
            part='phase',
            extension=['nii', 'nii.gz'],
            run=run
        )

    def get_b0_magnitude1(self, subject, session, run=None,
                          allow_multiple_files=False):
        b_maps_run_id = 1 if run is None else (run - 1) * 2 + 1
        return self.get_file(
            subject=subject,
            session=session,
            run=b_maps_run_id,
            acquisition="dznebnB0",
            suffix="magnitude1",
            extension=['nii', 'nii.gz']
        )

    def get_b0_magnitude2(self, subject, session, run=None,
                          allow_multiple_files=False):
        b_maps_run_id = 1 if run is None else (run - 1) * 2 + 1
        return self.get_file(
            subject=subject,
            session=session,
            run=b_maps_run_id,
            acquisition="dznebnB0",
            suffix="magnitude2",
            extension=['nii', 'nii.gz']
        )

    def get_b1_lte(self, subject, session, run=None,
                   allow_multiple_files=False):
        return self.get_file(
            subject=subject,
            session=session,
            suffix='T2w',
            part='phase',
            extension=['nii', 'nii.gz'],
            run=run
        )

    def get_b1_map_siemens(self, subject, session, run=None, extension='nii.gz',
                           allow_multiple_files=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B1Mape2",
            suffix="TB1map", extension=extension, datatype='fmap')
        return self.get_file(**entities)

    def get_b1_ste(self, subject, session, run=None,
                   allow_multiple_files=False):
        return self.get_file(
            subject=subject,
            session=session,
            run=run,
            acquisition="dznebnB1",
            suffix="magnitude1",
            extension=['nii', 'nii.gz']
        )

    def get_b1_fid(self, subject, session, run=None,
                   allow_multiple_files=False):
        return self.get_file(
            subject=subject,
            session=session,
            run=run,
            acquisition="dznebnB1",
            suffix="magnitude2",
            extension=['nii', 'nii.gz']
        )

    def prepare_data_workflow(self, base_dir=os.getcwd(), name="prepare_data",
                              subject=None, session=None, run=None):
        sub_ses_run_combinations = self.get_subject_session_run_combinations(
            subject=subject, session=session, run=run)

        inputs = []
        # outputs = []
        for combination in sub_ses_run_combinations:
            b0_map_siemens_file = self.get_b0_map_siemens(**combination)
            b0_te_delta = b0_map_siemens_file.entities["EchoTime2"] - \
                          b0_map_siemens_file.entities["EchoTime1"]

            input_dict = dict(
                subject=combination["subject"],
                session=combination["session"],
                run=combination["run"],
                b0_map_siemens_file=b0_map_siemens_file,
                b0_te_delta=b0_te_delta,
                b0_map_percent_file=self.get_b0_map_radian(**combination,
                                                           generate=True),
                b0_magnitude1_file=self.get_b0_magnitude1(**combination),
                b0_anat_ref_file=self.get_b0_anat_ref(**combination,
                                                      generate=True),
                b1_ste_file=self.get_b1_ste(**combination),
                b1_fid_file=self.get_b1_fid(**combination),
                b1_anat_ref_file=self.get_b1_anat_ref(**combination,
                                                      generate=True),
                b1_map_siemens_file=self.get_b1_map_siemens(**combination),
                b1_map_percent_file=self.get_b1_in_percent(**combination,
                                                           generate=True),
                axis_wrap_around=1,
                n_voxels_wrap_around=47,
                fa_b1_in_degrees=60,
                fa_nominal_in_degrees=20,
                pulse_duration_in_seconds=2.46e-3
            )
            inputs.append(input_dict)

        wf = pe.Workflow(name=name)
        wf.base_dir = base_dir

        # set up bids input node
        input_node = Node(IdentityInterface(fields=list(inputs[0].keys())),
                          name='bids_input_node')
        keys = inputs[0].keys()
        input_node.iterables = [
            (key, [input_dict[key] for input_dict in inputs]) for key in keys]
        input_node.synchronize = True

        # unwrap b0 map
        unwrap_phase_b0_node = pe.Node(interface=util.Function(
            input_names=['b0_phase_diff_file', 'b0_te_delta'],
            output_names=['out_file'],
            function=unwrap_phase_b0_siemens),
            name='unwrap_phase_b0')
        wf.connect(input_node, 'b0_map_siemens_file', unwrap_phase_b0_node,
                   'b0_phase_diff_file')
        wf.connect(input_node, 'b0_te_delta', unwrap_phase_b0_node,
                   'b0_te_delta')

        compute_b1_ref_node = pe.Node(interface=util.Function(
            input_names=['b1_ste_file', 'b1_fid_file'],
            output_names=['out_file'],
            function=compute_b1_anat_ref_from_lte_ste),
            name='compute_b1_ref')
        wf.connect(input_node, 'b1_ste_file', compute_b1_ref_node,
                   'b1_ste_file')
        wf.connect(input_node, 'b1_fid_file', compute_b1_ref_node,
                   'b1_fid_file')

        correct_phase_wrap_around_wf = correct_phase_wrap_around_workflow()
        wf.connect(compute_b1_ref_node, "out_file",
                   correct_phase_wrap_around_wf, "input_node.b1_anat_ref_file")
        wf.connect(input_node, "b1_map_siemens_file",
                   correct_phase_wrap_around_wf, "input_node.b1_map_file")
        wf.connect(input_node, "axis_wrap_around",
                   correct_phase_wrap_around_wf, "input_node.axis")
        wf.connect(input_node, "n_voxels_wrap_around",
                   correct_phase_wrap_around_wf, "input_node.n_voxels")

        copy_b1_map = Node(Function(
            input_names=["in_file", "target"],
            output_names=["copied_file"],
            function=simple_copy),
            name="copy_b1_map"
        )
        correct_phase_wrap_around_wf_output_node = correct_phase_wrap_around_wf.get_node(
            "output_node")
        wf.connect(correct_phase_wrap_around_wf_output_node, "b1_map_file",
                   copy_b1_map, "in_file")
        wf.connect(input_node, "b1_map_percent_file",
                   copy_b1_map, "target")

        copy_b1_anat_ref = Node(Function(
            input_names=["in_file", "target"],
            output_names=["copied_file"],
            function=simple_copy),
            name="copy_b1_anat_ref"
        )
        wf.connect(correct_phase_wrap_around_wf_output_node, "b1_anat_ref_file",
                   copy_b1_anat_ref, "in_file")
        wf.connect(input_node, "b1_anat_ref_file", copy_b1_anat_ref, "target")

        copy_b0_map = Node(Function(
            input_names=["in_file", "target"],
            output_names=["copied_file"],
            function=simple_copy),
            name="copy_b0_map"
        )
        wf.connect(unwrap_phase_b0_node, "out_file", copy_b0_map, "in_file")
        wf.connect(input_node, "b0_map_percent_file", copy_b0_map, "target")

        correct_b1_with_b0_wf = correct_b1_with_b0()
        wf.connect(unwrap_phase_b0_node, "out_file",
                   correct_b1_with_b0_wf, "input_node.b0_map_file")
        wf.connect(correct_phase_wrap_around_wf_output_node, "b1_map_file",
                   correct_b1_with_b0_wf, "input_node.b1_map_file")
        wf.connect(input_node, "b0_magnitude1_file",
                   correct_b1_with_b0_wf, "input_node.b0_anat_ref_file")
        wf.connect(correct_phase_wrap_around_wf_output_node, "b1_anat_ref_file",
                   correct_b1_with_b0_wf, "input_node.b1_anat_ref_file")
        wf.connect(input_node, "fa_b1_in_degrees",
                   correct_b1_with_b0_wf, "input_node.fa_b1_in_degrees")
        wf.connect(input_node, "fa_nominal_in_degrees",
                   correct_b1_with_b0_wf, "input_node.fa_nominal_in_degrees")
        wf.connect(input_node, "pulse_duration_in_seconds",
                   correct_b1_with_b0_wf, "input_node.pulse_duration_in_seconds")

        # input_node = pe.Node(util.IdentityInterface(
        #     fields=['b0_map_file',
        #             'b1_map_file',
        #             'b0_anat_ref_file',
        #             'b1_anat_ref_file',
        #             'fa_b1_in_degrees',
        #             'fa_nominal_in_degrees',
        #             'pulse_duration_in_seconds']),
        #     name='input_node')
        # output_node = pe.Node(util.IdentityInterface(fields=['out_file']),
        #                       name='output_node')


        return wf

    def preprocess_relaxation_images_workflow(self, base_dir=os.getcwd(),
                                              name="preprocess_relaxation_images",
                                              subject=None, session=None,
                                              run=None):
        pass

    def estimate_relaxation_maps_workflow(self, base_dir=os.getcwd(),
                                          name="estimate_relaxation_maps",
                                          subject=None, session=None, run=None):
        pass


def compute_b1_anat_ref_from_lte_ste(b1_ste_file, b1_fid_file):
    import nibabel as nib
    import os

    base_dir = os.getcwd()

    # read ste and fid b1 magnitude images
    b1_ste_file_nib = nib.load(b1_ste_file)
    b1_fid_image_nib = nib.load(b1_fid_file)
    b1_ste_image = b1_ste_file_nib.get_fdata()
    b1_fid_image = b1_fid_image_nib.get_fdata()

    # Compute the B1 ref
    b1_ref = (2 * b1_ste_image + b1_fid_image)

    # write b1 map
    b1_output_filename = os.path.join(base_dir, 'b1ref.nii.gz')
    b1_image_nib = nib.Nifti1Image(b1_ref, b1_ste_file_nib.affine,
                                   b1_ste_file_nib.header)
    nib.save(b1_image_nib, b1_output_filename)

    return b1_output_filename


def correct_phase_wrap_around_workflow(base_dir=os.getcwd(),
                                       name="correct_phase_wrap_around"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = Node(
        IdentityInterface(
            fields=['b1_anat_ref_file',
                    'b1_map_file',
                    'axis',
                    'n_voxels']),
        name='input_node'
    )
    output_node = pe.Node(util.IdentityInterface(
        fields=['b1_anat_ref_file',
                'b1_map_file'
                'brain_mask_file',
                'untouched_mask_file']),
        name='output_node')

    cut_and_merge_b1_map = Node(Function(
        input_names=['in_file', 'n_voxels', 'axis'],
        output_names=['out_file', 'untouched_mask_file'],
        function=cut_and_merge_image),
        name='cut_and_merge_b1_map')
    wf.connect(input_node, "b1_map_file",
               cut_and_merge_b1_map, "in_file")
    wf.connect(input_node, "axis",
               cut_and_merge_b1_map, "axis")
    wf.connect(input_node, "n_voxels",
               cut_and_merge_b1_map, "n_voxels")

    cut_and_merge_b1_anat_ref = Node(Function(
        input_names=['in_file', 'n_voxels', 'axis'],
        output_names=['out_file', 'untouched_mask_file'],
        function=cut_and_merge_image),
        name='cut_and_merge_b1_anat_ref')
    wf.connect(input_node, "b1_anat_ref_file",
               cut_and_merge_b1_anat_ref, "in_file")
    wf.connect(input_node, "axis",
               cut_and_merge_b1_anat_ref, "axis")
    wf.connect(input_node, "n_voxels",
               cut_and_merge_b1_anat_ref, "n_voxels")

    create_brain_mask_node = Node(Function(
        input_names=['in_file', 'fwhm'],
        output_names=['out_file'],
        function=create_brain_mask_from_anatomical_b1),
        name='create_brain_mask')
    create_brain_mask_node.inputs.fwhm = 8
    wf.connect(cut_and_merge_b1_anat_ref, "out_file",
               create_brain_mask_node, "in_file")

    impaint_b1_map = Node(Function(
        input_names=['in_file', 'brain_mask_file', 'fwhm', 'iterations'],
        output_names=['out_file'],
        function=inpaint),
        name='impaint_b1_map')
    impaint_b1_map.inputs.fwhm = 2
    impaint_b1_map.inputs.iterations = 15
    wf.connect(cut_and_merge_b1_map, "out_file",
               impaint_b1_map, "in_file")
    wf.connect(create_brain_mask_node, "out_file",
               impaint_b1_map, "brain_mask_file")

    impaint_b1_anat_ref = Node(Function(
        input_names=['in_file', 'brain_mask_file', 'fwhm', 'iterations'],
        output_names=['out_file'],
        function=inpaint),
        name='impaint_b1_anat_ref')
    impaint_b1_anat_ref.inputs.fwhm = 2
    impaint_b1_anat_ref.inputs.iterations = 10
    wf.connect(cut_and_merge_b1_anat_ref, "out_file",
               impaint_b1_anat_ref, "in_file")
    wf.connect(create_brain_mask_node, "out_file",
               impaint_b1_anat_ref, "brain_mask_file")

    # set outputs
    wf.connect(cut_and_merge_b1_map, "untouched_mask_file",
               output_node, "untouched_mask_file")
    wf.connect(create_brain_mask_node, "out_file",
               output_node, "brain_mask_file")
    wf.connect(impaint_b1_map, "out_file",
               output_node, "b1_map_file")
    wf.connect(impaint_b1_anat_ref, "out_file",
               output_node, "b1_anat_ref_file")

    return wf
