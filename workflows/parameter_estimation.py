from nipype.interfaces.utility import IdentityInterface
import nipype.pipeline.engine as pe
from nipype import Node, Function
import nipype.interfaces.fsl as fsl
import os
from nodes.processing import QiJsr
from utils.processing import compute_t2_t1_amplitude_maps
from nipype.interfaces.utility import Merge
from utils.io import write_json


def estimate_relaxation_ssfp(base_dir=os.getcwd(),
                             name="estimate_relaxation_ssfp"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = pe.Node(
        IdentityInterface(fields=[
            "b1_map_file",
            "t1w_files",
            "t2w_files",
            "brain_mask_file",
            "qi_jsr_config_dict"
        ]),
        name="input_node"
    )

    output_node = pe.Node(
        IdentityInterface(fields=[
            "r1_map_file",
            "r2_map_file",
            "t1_map_file",
            "t2_map_file",
        ]),
        name="output_node"
    )

    # merge t1w images
    t1w_image_merge = Node(fsl.Merge(dimension='t'), name='t1w_image_merge')
    wf.connect(input_node, 't1w_files', t1w_image_merge, 'in_files')

    # merge t2w images
    t2w_image_merge = Node(fsl.Merge(dimension='t'), name='t2w_image_merge')
    wf.connect(input_node, 't2w_files', t2w_image_merge, 'in_files')

    # Create a Function node to write the JSON file
    write_json_node = Node(
        Function(
            input_names=["data", "filename"],
            output_names=["out_file"],
            function=write_json
        ),
        name="write_json_node"
    )
    write_json_node.inputs.filename = "qi_jsr_config.json"
    wf.connect(input_node, 'qi_jsr_config_dict', write_json_node, 'data')

    # run iq jsr
    qi_jsr = pe.Node(QiJsr(npsi=6), name="qi_jsr")
    wf.connect(write_json_node, 'out_file', qi_jsr, 'json_file')
    wf.connect(t1w_image_merge, "merged_file", qi_jsr, "spgr_file")
    wf.connect(t2w_image_merge, "merged_file", qi_jsr, "ssfp_file")
    wf.connect(input_node, "b1_map_file", qi_jsr, "b1_file")
    wf.connect(input_node, "brain_mask_file", qi_jsr, "mask_file")

    # compute R1 map
    compute_r1 = pe.Node(
        fsl.ImageMaths(op_string='-recip'),
        name="compute_r1")
    wf.connect(qi_jsr, "t1_map_file",
               compute_r1, "in_file")

    # compute R2 map
    compute_r2 = pe.Node(
        fsl.ImageMaths(op_string='-recip'),
        name="compute_r2")
    wf.connect(qi_jsr, "t2_map_file",
               compute_r2, "in_file")

    wf.connect(compute_r1, "out_file", output_node, "r1_map_file")
    wf.connect(compute_r2, "out_file", output_node, "r2_map_file")
    wf.connect(qi_jsr, "t1_map_file", output_node, "t1_map_file")
    wf.connect(qi_jsr, "t2_map_file", output_node, "t2_map_file")

    return wf


def estimate_relaxation_3d_epi(base_dir=os.getcwd(),
                             name="estimate_relaxation_3d_epi"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = pe.Node(
        IdentityInterface(fields=[
            "t2w_magnitude_file",
            "t2w_phase_file",
            "brain_mask_file",
            "b1_map_file",
            "rf_phase_increments",
            "flip_angle",
            "repetition_time"
        ]),
        name="input_node"
    )
    input_node.inputs.brain_mask_file = None

    output_node = pe.Node(
        IdentityInterface(fields=[
            "r1_map_file",
            "r2_map_file",
            "t1_map_file",
            "t2_map_file",
            "am_map_file"
        ]),
        name="output_node"
    )

    # scale B1 map to percent
    scale_b1_to_percent = pe.Node(
        fsl.ImageMaths(op_string='-mul 100.0'),
        name="scale_b1_to_percent")
    wf.connect(input_node, "b1_map_file",
               scale_b1_to_percent, "in_file")

    # compute T1, T2, AM
    compute_t2_t1_am_node = Node(
        Function(input_names=["magnitude_file", "phase_file", "mask_file",
                              "b1_map_file", "repetition_time",
                              "flip_angle",
                              "rf_phase_increments"],
                 output_names=["t2_map_file", "t1_map_file", "am_map_file"],
                 function=compute_t2_t1_amplitude_maps),
        name="compute_t2_t1_am")
    wf.connect(input_node, "brain_mask_file",
               compute_t2_t1_am_node, "mask_file")
    wf.connect(input_node, "t2w_magnitude_file",
               compute_t2_t1_am_node, "magnitude_file")
    wf.connect(input_node, "t2w_phase_file",
               compute_t2_t1_am_node, "phase_file")
    wf.connect(input_node, "rf_phase_increments",
               compute_t2_t1_am_node, "rf_phase_increments")
    wf.connect(scale_b1_to_percent, "out_file",
               compute_t2_t1_am_node, "b1_map_file")
    wf.connect(input_node, "flip_angle",
               compute_t2_t1_am_node, "flip_angle")
    wf.connect(input_node, "repetition_time",
               compute_t2_t1_am_node, "repetition_time")

    # compute R1 map
    compute_r1 = pe.Node(
        fsl.ImageMaths(op_string='-recip'),
        name="compute_r1")
    wf.connect(compute_t2_t1_am_node, "t1_map_file",
               compute_r1, "in_file")

    # compute R2 map
    compute_r2 = pe.Node(
        fsl.ImageMaths(op_string='-recip'),
        name="compute_r2")
    wf.connect(compute_t2_t1_am_node, "t2_map_file",
               compute_r2, "in_file")

    wf.connect(compute_r1, "out_file", output_node, "r1_map_file")
    wf.connect(compute_r2, "out_file", output_node, "r2_map_file")
    wf.connect(compute_t2_t1_am_node, "t1_map_file", output_node, "t1_map_file")
    wf.connect(compute_t2_t1_am_node, "t2_map_file", output_node, "t2_map_file")
    wf.connect(compute_t2_t1_am_node, "am_map_file", output_node, "am_map_file")

    return wf