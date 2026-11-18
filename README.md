# t2_mapping_traveling_heads

## How to setup
### This package
To clone and setup repository run
```bash
git clone https://github.com/quantitative-mri-and-in-vivo-histology/t2_mapping_traveling_heads.git
git submodule update --init --recursive
```
After installing EPGpp (see below):
```bash
pip install /external/MagPhsT2
```
To use the faster t2greps fitting backend (optional, see `--fit_backend` below):
```bash
pip install external/t2greps
```

### Python packages
To install python dependencies run
```bash
pip install -r requirements.txt
```

### External dependencies
Please install also the following external dependencies:
- QUIT (only for SSFP/SPGR, see https://github.com/spinicist/QUIT)
- ANTs (see https://github.com/ANTsX/ANTs)
- EPGpp (see https://github.com/mrphysics-bonn/EPGpp)

## How to run
### 3D-EPI Processing
Run
```bash
python -m cli.process_3depi \
    --t2w_mag T2w_mag.nii.gz --t2w_phase T2w_phase.nii.gz \
    --b1_map B1_map.nii.gz --b1_anat_ref B1_anat_ref.nii.gz \
    --t1w T1w.nii.gz --output_dir out/ [--fit_backend t2greps]
```

| Parameter | Required | Description |
|---|---|---|
| `--t2w_mag` | yes | T2-weighted magnitude image. |
| `--t2w_phase` | yes | T2-weighted phase image in radians. |
| `--b1_map` | yes | B1 map normalized to 1 (1 = perfectly homogeneous field). |
| `--b1_anat_ref` | yes | B1 anatomical reference: a magnitude image aligned with the B1 map. |
| `--output_dir` | yes | Directory to store the outputs. |
| `--echo_time` | no | Echo time in seconds. Default: `EchoTime` from the JSON file of `--t2w_mag`. |
| `--repetition_time` | no | Repetition time in seconds. Default: `RepetitionTimeExcitation` from the JSON file of `--t2w_mag`. |
| `--flip_angle` | no | Flip angle in degrees. Default: `FlipAngle` from the JSON file of `--t2w_mag`. |
| `--rf_phase_increments` | no | Positive RF phase increments in degrees, e.g. `1 2 4 1.5 3 5`. Default: `RfPhaseIncrement` from the JSON file of `--t2w_mag`. The T2w images must have 2 × the number of increments volumes, ordered `[+inc1, -inc1, +inc2, -inc2, ...]`. |
| `--t1w` | no | T1w image for brain mask extraction. It is registered to the T2w magnitude image first. Mutually exclusive with `--mask`. |
| `--mask` | no | Binary brain mask aligned with the T2w magnitude image, used directly. Mutually exclusive with `--t1w`. Without `--t1w` or `--mask`, all voxels are fitted. |
| `--fit_backend` | no | `magphst2` (default) or `t2greps`. [t2greps](https://github.com/quantitative-mri-and-in-vivo-histology/t2greps) computes the same fit in seconds to minutes instead of hours. |
| `--preprocess_only` | no | Only run the preprocessing. |
| `--n_procs` | no | Number of processors (default: all available cores). |

### SSFP/SPGR-based Processing
Run
```bash
python -m cli.process_ssfp_spgr \
    --t1w_files spgr_1.nii.gz spgr_2.nii.gz --t2w_files ssfp_1.nii.gz ssfp_2.nii.gz \
    --b1_map_file B1_map.nii.gz --b1_anat_ref_file B1_anat_ref.nii.gz --output_dir out/
```

| Parameter | Required | Description |
|---|---|---|
| `--t1w_files` | yes | T1-weighted (SPGR) images. The first is the reference for motion correction. Each needs a JSON file with the same name containing `FlipAngle`, `RepetitionTimeExcitation`, `RfPulseDuration` and `RfPhaseIncrement`. |
| `--t2w_files` | yes | T2-weighted (SSFP) images. The first is the reference for motion correction. Each needs a JSON file with the same name containing `FlipAngle`, `RepetitionTimeExcitation` and `EchoTime`. |
| `--b1_map_file` | yes | B1 map normalized to 1 (1 = perfectly homogeneous field). |
| `--b1_anat_ref_file` | yes | B1 anatomical reference. |
| `--output_dir` | yes | Directory to store the outputs. |
| `--temp_dir` | no | Directory for temporary outputs (default: current directory). |
| `--preprocess_only` | no | Only run the preprocessing. |
| `--n_procs` | no | Number of processors (default: all available cores). |
