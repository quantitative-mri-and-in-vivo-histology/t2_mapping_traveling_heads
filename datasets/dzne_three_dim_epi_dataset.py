import os
import math
from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
from nipype import Node, Function
from datasets.three_dim_epi_dataset import ThreeDimEpiDataset
from utils.processing import unwrap_phase_b0_siemens, cut_and_merge_image, \
    create_brain_mask_from_anatomical_b1, inpaint, \
    compute_b1_magnitude_image_from_ste_lte
from utils.io import ExplicitPathDataSink
from workflows.fieldmap_workflows import correct_b1_with_b0
import nipype.interfaces.fsl as fsl


class DzneThreeDimEpiDataset(ThreeDimEpiDataset):

    def __init__(self, bids_root, derivatives_output_folder, derivatives=None):
        # Call the constructor of the parent class (3dEpiBids)
        super().__init__(bids_root=bids_root,
                         derivatives_output_folder=derivatives_output_folder,
                         derivatives=derivatives)

    def get_t2w_phase_raw_siemens(self, subject, session, run=None,
                                  allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, desc=None,
            suffix='T2w', part='phase', extension=self._nifti_read_ext,
            datatype='anat')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b0_map_siemens(self, subject, session, run=None,
                           allow_multiple_files=False, generate=False):
        b_maps_run_id = 1 if run is None else (run - 1) * 2 + 1
        entities = dict(
            subject=subject, session=session, run=b_maps_run_id,
            acquisition="dznebnB0",
            suffix='phase2', extension=self._nifti_read_ext, datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b0_magnitude1(self, subject, session, run=None,
                          allow_multiple_files=False, generate=False):
        b_maps_run_id = 1 if run is None else (run - 1) * 2 + 1
        entities = dict(
            subject=subject, session=session, run=b_maps_run_id,
            acquisition="dznebnB0",
            suffix='magnitude1', extension=self._nifti_read_ext,
            datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b0_magnitude2(self, subject, session, run=None,
                          allow_multiple_files=False, generate=False):
        b_maps_run_id = 1 if run is None else (run - 1) * 2 + 1
        entities = dict(
            subject=subject, session=session, run=b_maps_run_id,
            acquisition="dznebnB0",
            suffix='magnitude2', extension=self._nifti_read_ext,
            datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b1_map_siemens(self, subject, session, run=None,
                           allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B1Mape2",
            suffix="TB1map", extension=self._nifti_read_ext, datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b1_ste(self, subject, session, run=None,
                   allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="dznebnB1",
            suffix='magnitude1', extension=self._nifti_read_ext,
            datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_b1_fid(self, subject, session, run=None,
                   allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="dznebnB1",
            suffix='magnitude2', extension=self._nifti_read_ext,
            datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def get_untouched_b1_mask(self, subject, session, run=None,
                              allow_multiple_files=False, generate=False):
        entities = dict(
            subject=subject, session=session, run=run, acquisition="B1",
            desc="untouched", suffix="mask", extension=self._nifti_read_ext,
            datatype='fmap')
        return self._get_file(entities,
                              allow_multiple_files=allow_multiple_files,
                              generate=generate)

    def prepare_data_workflow(self, base_dir=os.getcwd(), name="prepare_data",
                              subject=None, session=None, run=None):
        sub_ses_run_combinations = self.get_subject_session_run_combinations(
            subject=subject, session=session, run=run)

        inputs = []
        for combination in sub_ses_run_combinations:
            b0_map_siemens_file = self.get_b0_map_siemens(**combination)
            b0_te_delta = b0_map_siemens_file.entities["EchoTime2"] - \
                          b0_map_siemens_file.entities["EchoTime1"]

            b1_map_siemens_file = self.get_b1_map_siemens(**combination)
            fa_b1_in_degrees = b1_map_siemens_file.entities["FlipAngle"]
            b1_scaling_factor = 1.0/(fa_b1_in_degrees*10)

            t2w_phase_raw_siemens = self.get_t2w_phase_raw_siemens(
                **combination)
            fa_nominal_in_degrees = t2w_phase_raw_siemens.entities["FlipAngle"]

            do_phase_wrap_around_correction = subject in ["phy002", "phy003",
                                                          "phy004"]

            input_dict = dict(
                subject=combination["subject"],
                session=combination["session"],
                run=combination["run"],
                b0_map_siemens_file=b0_map_siemens_file,
                b0_te_delta=b0_te_delta,
                b0_map_radian_file=self.get_b0_map_radian(**combination,
                                                          generate=True),
                b0_magnitude1_file=self.get_b0_magnitude1(**combination),
                b0_anat_ref_file=self.get_b0_anat_ref(**combination,
                                                      generate=True),
                b1_ste_file=self.get_b1_ste(**combination),
                b1_fid_file=self.get_b1_fid(**combination),
                b1_anat_ref_file=self.get_b1_anat_ref(**combination,
                                                      generate=True),
                b1_map_siemens_file=b1_map_siemens_file,
                b1_plus_relative_unadjusted_map_file=self.get_b1_plus_relative_unadjusted_map(
                    **combination,
                    generate=True),
                b1_plus_relative_t2w_adjusted_map_file=self.get_b1_plus_relative_t2w_adjusted_map(
                    **combination,
                    generate=True),
                axis_wrap_around=1,
                n_voxels_wrap_around=47,
                do_phase_wrap_around_correction=do_phase_wrap_around_correction,
                fa_b1_in_degrees=fa_b1_in_degrees,
                fa_nominal_in_degrees=fa_nominal_in_degrees,
                b1_scaling_factor=b1_scaling_factor,
                pulse_duration_in_seconds=2.46e-3,
                t2w_phase_raw_siemens=self.get_t2w_phase_raw_siemens(
                    **combination),
                t2w_phase_radian_raw=self.get_t2w_phase_radian_raw(
                    **combination,
                    generate=True),
                untouched_b1_mask=self.get_untouched_b1_mask(**combination,
                                                             generate=True)
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

        # scale phase to radian
        scaling_factor = math.pi / 4096.0
        scale_phase_from_siemens_to_radian = pe.Node(
            fsl.ImageMaths(op_string='-mul {}'.format(scaling_factor)),
            name="scale_phase_from_siemens_to_rad")
        wf.connect(input_node, "t2w_phase_raw_siemens",
                   scale_phase_from_siemens_to_radian, "in_file")

        # scale phase to radian
        normalize_b1 = pe.Node(
            fsl.ImageMaths(op_string='-mul {}'.format(b1_scaling_factor)),
            name="normalize_b1")
        wf.connect(input_node, "b1_map_siemens_file",
                   normalize_b1, "in_file")

        # convert B0 map to radian
        unwrap_phase_b0_node = pe.Node(interface=util.Function(
            input_names=['b0_phase_diff_file', 'b0_te_delta'],
            output_names=['out_file'],
            function=unwrap_phase_b0_siemens),
            name='unwrap_phase_b0')
        wf.connect(input_node, 'b0_map_siemens_file', unwrap_phase_b0_node,
                   'b0_phase_diff_file')
        wf.connect(input_node, 'b0_te_delta', unwrap_phase_b0_node,
                   'b0_te_delta')

        # compute B1 anatomical reference image
        compute_b1_ref_node = pe.Node(interface=util.Function(
            input_names=['b1_ste_file', 'b1_fid_file'],
            output_names=['out_file'],
            function=compute_b1_magnitude_image_from_ste_lte),
            name='compute_b1_ref')
        wf.connect(input_node, 'b1_ste_file', compute_b1_ref_node,
                   'b1_ste_file')
        wf.connect(input_node, 'b1_fid_file', compute_b1_ref_node,
                   'b1_fid_file')

        correct_phase_wrap_around_wf = self._correct_phase_wrap_around_workflow()
        wf.connect(compute_b1_ref_node, "out_file",
                   correct_phase_wrap_around_wf, "input_node.b1_anat_ref_file")
        wf.connect(normalize_b1, "out_file",
                   correct_phase_wrap_around_wf, "input_node.b1_map_file")
        wf.connect(input_node, "axis_wrap_around",
                   correct_phase_wrap_around_wf, "input_node.axis")
        wf.connect(input_node, "n_voxels_wrap_around",
                   correct_phase_wrap_around_wf, "input_node.n_voxels")

        # b1 adjustment for T2w images
        correct_b1_with_b0_wf = correct_b1_with_b0()
        wf.connect(unwrap_phase_b0_node, "out_file",
                   correct_b1_with_b0_wf, "input_node.b0_map_file")
        wf.connect(correct_phase_wrap_around_wf, "output_node.b1_map_file",
                   correct_b1_with_b0_wf, "input_node.b1_map_file")
        wf.connect(input_node, "b0_magnitude1_file",
                   correct_b1_with_b0_wf, "input_node.b0_anat_ref_file")
        wf.connect(correct_phase_wrap_around_wf, "output_node.b1_anat_ref_file",
                   correct_b1_with_b0_wf, "input_node.b1_anat_ref_file")
        wf.connect(input_node, "fa_b1_in_degrees",
                   correct_b1_with_b0_wf, "input_node.fa_b1_in_degrees")
        wf.connect(input_node, "fa_nominal_in_degrees",
                   correct_b1_with_b0_wf, "input_node.fa_nominal_in_degrees")
        wf.connect(input_node, "pulse_duration_in_seconds",
                   correct_b1_with_b0_wf,
                   "input_node.pulse_duration_in_seconds")

        # write B0 map in radian
        b0_anat_ref_data_sink = Node(
            ExplicitPathDataSink(), name='b0_anat_ref_data_sink')
        wf.connect([(input_node, b0_anat_ref_data_sink, [
            ("b0_magnitude1_file", "in_file"),
            ("b0_anat_ref_file", "out_file")
        ])])

        # write B0 anatomical reference image
        b0_map_data_sink = Node(
            ExplicitPathDataSink(), name='b0_map_data_sink')
        wf.connect(unwrap_phase_b0_node, "out_file",
                   b0_map_data_sink, "in_file")
        wf.connect(input_node, "b0_map_radian_file",
                   b0_map_data_sink, "out_file")

        # write B1 untouched mask
        b1_untouched_mask_data_sink = Node(
            ExplicitPathDataSink(), name='b1_untouched_mask_data_sink')
        wf.connect(correct_phase_wrap_around_wf,
                   "output_node.untouched_mask_file",
                   b1_untouched_mask_data_sink, "in_file")
        wf.connect(input_node, "untouched_b1_mask",
                   b1_untouched_mask_data_sink, "out_file")

        # write B1 anatomical reference image
        b1_anat_ref_data_sink = Node(
            ExplicitPathDataSink(), name='b1_anat_ref_data_sink')
        wf.connect(correct_phase_wrap_around_wf, "output_node.b1_anat_ref_file",
                   b1_anat_ref_data_sink, "in_file")
        wf.connect(input_node, "b1_anat_ref_file",
                   b1_anat_ref_data_sink, "out_file")

        b1_unadjusted_map_data_sink = Node(
            ExplicitPathDataSink(), name='b1_unadjusted_map_data_sink')
        wf.connect(correct_phase_wrap_around_wf, "output_node.b1_map_file",
                   b1_unadjusted_map_data_sink, "in_file")
        wf.connect(input_node, "b1_plus_relative_unadjusted_map_file",
                   b1_unadjusted_map_data_sink, "out_file")

        b1_t2w_adjusted_map_data_sink = Node(
            ExplicitPathDataSink(), name='b1_t2w_adjusted_map_data_sink')
        wf.connect(correct_b1_with_b0_wf, "output_node.out_file",
                   b1_t2w_adjusted_map_data_sink, "in_file")
        wf.connect(input_node, "b1_plus_relative_t2w_adjusted_map_file",
                   b1_t2w_adjusted_map_data_sink, "out_file")

        t2w_phase_radian_data_sink = Node(
            ExplicitPathDataSink(), name='t2w_phase_radian_data_sink')
        wf.connect(scale_phase_from_siemens_to_radian, "out_file",
                   t2w_phase_radian_data_sink, "in_file")
        wf.connect(input_node, "t2w_phase_radian_raw",
                   t2w_phase_radian_data_sink, "out_file")

        return wf

    def _correct_phase_wrap_around_workflow(self, base_dir=os.getcwd(),
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
                    'b1_map_file',
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

        inpaint_b1_map = Node(Function(
            input_names=['in_file', 'brain_mask_file', 'fwhm', 'iterations'],
            output_names=['out_file'],
            function=inpaint),
            name='impaint_b1_map')
        inpaint_b1_map.inputs.fwhm = 2
        inpaint_b1_map.inputs.iterations = 15
        wf.connect(cut_and_merge_b1_map, "out_file",
                   inpaint_b1_map, "in_file")
        wf.connect(create_brain_mask_node, "out_file",
                   inpaint_b1_map, "brain_mask_file")

        inpaint_b1_anat_ref = Node(Function(
            input_names=['in_file', 'brain_mask_file', 'fwhm', 'iterations'],
            output_names=['out_file'],
            function=inpaint),
            name='impaint_b1_anat_ref')
        inpaint_b1_anat_ref.inputs.fwhm = 2
        inpaint_b1_anat_ref.inputs.iterations = 10
        wf.connect(cut_and_merge_b1_anat_ref, "out_file",
                   inpaint_b1_anat_ref, "in_file")
        wf.connect(create_brain_mask_node, "out_file",
                   inpaint_b1_anat_ref, "brain_mask_file")

        # set outputs
        wf.connect(cut_and_merge_b1_map, "untouched_mask_file",
                   output_node, "untouched_mask_file")
        wf.connect(create_brain_mask_node, "out_file",
                   output_node, "brain_mask_file")
        wf.connect(inpaint_b1_map, "out_file",
                   output_node, "b1_map_file")
        wf.connect(inpaint_b1_anat_ref, "out_file",
                   output_node, "b1_anat_ref_file")

        return wf
