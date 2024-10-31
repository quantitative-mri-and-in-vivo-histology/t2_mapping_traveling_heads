# Define the default output pattern (unchanged)
DEFAULT_BIDS_OUTPUT_PATTERN = 'sub-{subject}/ses-{session}/{datatype}/' \
                              'sub-{subject}_ses-{session}[_acq-{acquisition}]' \
                              '[_run-{run}][_desc-{desc}][_part-{part}]_{suffix}.{extension}'

DEFAULT_NIFTI_READ_EXT_ENTITY = dict(extension=[".nii.gz", ".nii"])
DEFAULT_NIFTI_WRITE_EXT_ENTITY = dict(extension=".nii.gz")

# Standardized entity overrides for different file types
STANDARDIZED_ENTITY_OVERRIDES_T1W = dict(suffix='T1w', part=None)
STANDARDIZED_ENTITY_OVERRIDES_T2W = dict(suffix='T2w', part=None)
STANDARDIZED_ENTITY_OVERRIDES_T2W_MAG = dict(suffix='T2w', part="mag")
STANDARDIZED_ENTITY_OVERRIDES_T2W_PHASE = dict(suffix='T2w', part="phase")
STANDARDIZED_ENTITY_OVERRIDES_B0_PHASEDIFF_MAP = dict(suffix='phasediff',
                                                      acquisition='B0')
STANDARDIZED_ENTITY_OVERRIDES_B0_ANAT_REF = dict(suffix='magnitude',
                                            acquisition='B0ref')
STANDARDIZED_ENTITY_OVERRIDES_B1_MAP = dict(suffix='B1map', acquisition='B1')
STANDARDIZED_ENTITY_OVERRIDES_B1_ANAT_REF = dict(suffix='magnitude',
                                            acquisition='B1ref')

PROCESSED_ENTITY_OVERRIDES_B1_MAP = STANDARDIZED_ENTITY_OVERRIDES_B1_MAP
PROCESSED_ENTITY_OVERRIDES_B1_ANAT_REF = STANDARDIZED_ENTITY_OVERRIDES_B1_ANAT_REF
PROCESSED_ENTITY_OVERRIDES_T1W = dict(suffix='T1w', part=None)
PROCESSED_ENTITY_OVERRIDES_T2W_MAG = dict(suffix='T2w', part="mag",
                                          desc="preproc")
PROCESSED_ENTITY_OVERRIDES_T2W_PHASE = dict(suffix='T2w', part="phase",
                                            desc="preproc")
PROCESSED_ENTITY_OVERRIDES_T2W = dict(suffix='T2w', part=None)
PROCESSED_ENTITY_OVERRIDES_REG_REF_IMAGE = dict(desc="regRef", part=None)
PROCESSED_ENTITY_OVERRIDES_BRAIN_MASK = dict(suffix='mask', desc="brain",
                                             acquisition=None)
PROCESSED_ENTITY_OVERRIDES_R1_MAP = dict(suffix='R1map', acquisition=None,
                                         part=None)
PROCESSED_ENTITY_OVERRIDES_R2_MAP = dict(suffix='R2map', acquisition=None,
                                         part=None)
PROCESSED_ENTITY_OVERRIDES_T1_MAP = dict(suffix='T1map', acquisition=None,
                                         part=None)
PROCESSED_ENTITY_OVERRIDES_T2_MAP = dict(suffix='T2map', acquisition=None,
                                         part=None)
PROCESSED_ENTITY_OVERRIDES_AM_MAP = dict(suffix='AMmap', acquisition=None,
                                         part=None)
PROCESSED_ENTITY_OVERRIDES_MTSAT_MAP = dict(suffix='MTsat', acquisition=None,
                                         part=None)
PROCESSED_ENTITY_OVERRIDES_PD_MAP = dict(suffix='PDmap', acquisition=None,
                                         part=None)

