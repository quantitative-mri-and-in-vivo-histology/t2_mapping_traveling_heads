# t2_mapping_traveling_heads

## How to setup
### This package
To clone and setup repository run
```bash
git clone https://github.com/quantitative-mri-and-in-vivo-histology/t2_mapping_traveling_heads.git
git submodule update --init --recursive
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
python -m cli.process_3depi ...
```
### SSFP/SPGR-based Processing
Run
```bash
python -m cli.process_ssfp_spgr ...
```

