from nipype.interfaces.utility import IdentityInterface
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import nipype.interfaces.mrtrix3 as mrtrix3
from nipype import Node, Function
import nipype.interfaces.fsl as fsl
import os
from utils.processing import QiTgv, QiJsr
from nipype_utils import ApplyXfm4D, get_common_parent_directory
from nipype.interfaces.utility import Merge
from workflows.preprocessing_workflows import create_brain_mask_workflow
from pathlib import Path
from utils.io import write_json


def estimate_relaxation_ssfp_multi_file(base_dir=os.getcwd(),
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


def estimate_relaxation_ssfp(base_dir=os.getcwd(),
                             name="estimate_relaxation_ssfp"):
    wf = pe.Workflow(name=name)
    wf.base_dir = base_dir

    input_node = pe.Node(
        IdentityInterface(fields=[
            "b1_map_file",
            "t1w_fa_2_file",
            "t1w_fa_13_file",
            "t2w_fa_12_rf_180_file",
            "t2w_fa_49_rf_0_file",
            "t2w_fa_49_rf_180_file",
            "brain_mask_file"
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

    # qi_jsr_config_dict = dict(
    #     SPGR=dict(
    #         TR=0.0062,
    #         TE=0.003,
    #         FA=[2, 13]
    #     ),
    #     SSFP=dict(
    #         TR=0.006,
    #         Trf=0.0013,
    #         FA=[12, 49, 49],
    #         PhaseInc=[180, 0, 180]
    #     )
    # )

    qi_jsr_config_dict = dict(
        SPGR=dict(
            TR=0.0062,
            TE=0.00351,
            FA=[2, 13]
        ),
        SSFP=dict(
            TR=0.006,
            Trf=0.0013,
            FA=[12, 49, 49],
            PhaseInc=[180, 0, 180]
        )
    )

    # merge t1w images
    t1w_list_merge = Node(Merge(2),
                          name='t1w_list_merge')  # Merge will take 3 inputs
    t1w_image_merge = Node(fsl.Merge(dimension='t'), name='t1w_image_merge')
    wf.connect(input_node, 't1w_fa_2_file', t1w_list_merge, 'in1')
    wf.connect(input_node, 't1w_fa_13_file', t1w_list_merge, 'in2')
    wf.connect(t1w_list_merge, 'out', t1w_image_merge, 'in_files')

    # merge t2w images
    t2w_list_merge = Node(Merge(3),
                          name='t2w_list_merge')  # Merge will take 3 inputs
    t2w_image_merge = Node(fsl.Merge(dimension='t'), name='t2w_image_merge')
    wf.connect(input_node, 't2w_fa_12_rf_180_file', t2w_list_merge, 'in1')
    wf.connect(input_node, 't2w_fa_49_rf_0_file', t2w_list_merge, 'in2')
    wf.connect(input_node, 't2w_fa_49_rf_180_file', t2w_list_merge, 'in3')
    wf.connect(t2w_list_merge, 'out', t2w_image_merge, 'in_files')

    # Create a Function node to write the JSON file
    write_json_node = Node(
        Function(
            input_names=["data", "filename"],
            output_names=["out_file"],
            function=write_json
        ),
        name="write_json_node"
    )

    write_json_node.inputs.data = qi_jsr_config_dict
    write_json_node.inputs.filename = "qi_jsr_config.json"

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
