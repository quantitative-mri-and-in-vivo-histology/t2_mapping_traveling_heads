# Define the default output pattern (unchanged)
DEFAULT_BIDS_OUTPUT_PATTERN = 'sub-{subject}/ses-{session}/{datatype}/' \
                 'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                 '[_run-{run}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'

DEFAULT_NIFTI_READ_EXT_ENTITY = dict(extension=[".nii.gz", ".nii"])
DEFAULT_NIFTI_WRITE_EXT_ENTITY = dict(extension=".nii.gz")

# Standardized entity overrides for different file types
STANDARDIZED_ENTITY_OVERRIDES_T1W = dict(suffix='T1w', part=None)
STANDARDIZED_ENTITY_OVERRIDES_T1W_MAG = dict(suffix='T1w', part="mag")
STANDARDIZED_ENTITY_OVERRIDES_T1W_PHASE = dict(suffix='T1w', part="phase")
STANDARDIZED_ENTITY_OVERRIDES_T2W = dict(suffix='T2w', part=None)
STANDARDIZED_ENTITY_OVERRIDES_T2W_MAG = dict(suffix='T2w', part="mag")
STANDARDIZED_ENTITY_OVERRIDES_T2W_PHASE = dict(suffix='T2w', part="phase")
STANDARDIZED_ENTITY_OVERRIDES_B0_PHASEDIFF_MAP = dict(suffix='phasediff', acquisition='B0')
STANDARDIZED_ENTITY_OVERRIDES_B0_REF = dict(suffix='magnitude', acquisition='B0Ref')
STANDARDIZED_ENTITY_OVERRIDES_B1_MAP = dict(suffix='B1Map', acquisition='B1')
STANDARDIZED_ENTITY_OVERRIDES_B1_REF = dict(suffix='magnitude', acquisition='B1Ref')
