from bids.layout import BIDSLayout
from datasets.dzne_three_dim_epi_dataset import DzneThreeDimEpiDataset
from datasets.dzne_three_dim_epi_dataset import ThreeDimEpiDataset

# class MyBidsHandler:
#     ENTITY_DEFINITIONS = {
#         'b0_map_radian': lambda subject, session, run=None,
#                                 extension='nii.gz': dict(
#             subject=subject, session=session, run=run, desc="radian",
#             suffix="B0map", extension=extension, datatype='fmap'
#         ),
#         'b1_anat_ref': lambda subject, session, run=None,
#                               extension='nii.gz': dict(
#             subject=subject, session=session, run=run, acquisition="B1Anat",
#             suffix="magnitude", extension=extension, datatype='fmap'
#         ),
#         # Add more definitions as needed...
#     }
#
#     def __init__(self):
#         # Automatically create methods for each entity definition
#         for name, entity_func in self.ENTITY_DEFINITIONS.items():
#             method_name = f'get_{name}'
#             method = self._create_get_method(entity_func)
#             setattr(self, method_name, method)
#
#     def _create_get_method(self, entity_func):
#         """Generates a method for retrieving or generating paths."""
#
#         def method(subject, session, run=None, generate=False,
#                    extension='nii.gz', allow_multiple_files=False):
#             entities = entity_func(subject, session, run=run,
#                                    extension=extension)
#             return self.get_something(entities,
#                                       allow_multiple_files=allow_multiple_files,
#                                       generate=generate)
#
#         return method
#
#     def get_something(self, entities, allow_multiple_files=False,
#                       generate=False):
#         """Handles the common logic of getting or generating a path."""
#         if generate:
#             return self.make_path(entities)
#         else:
#             return self.get_file(**entities,
#                                  allow_multiple_files=allow_multiple_files)
#
#     # Placeholder methods for demonstration (replace with actual implementations)
#     def make_path(self, entities):
#         return f"Generated path with entities: {entities}"
#
#     def get_file(self, **entities):
#         return f"Retrieved file with entities: {entities}"



if __name__ == "__main__":

    bids_root = "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids"
    derivatives = ["/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/derivatives/SIEMENS",
                   "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/derivatives/FieldMapCorrection"]
    out_derivatives_folder = "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/derivatives/nipype"

    dataset = DzneThreeDimEpiDataset(bids_root, out_derivatives_folder, derivatives)

    #
    # layout = BIDSLayout(bids_root, out_derivatives_folder, derivatives=derivatives, validate=False)

    print(dataset.get_b1_anat_ref(subject="phy003", session="001", run=None, generate=True))
    print(dataset.get_b0_map_radian(subject="phy003", session="001", run=None, generate=True))
    print(dataset.get_b1_plus_relative_t2w_adjusted_map(subject="phy001", session="001", run=1, generate=False))

    # print(dataset.get_t2w_phase_raw("phy001", "001", 1))
    # print(dataset.get_corrected_b1_map("phy001", "001", 1, generate=True))
    # print(dataset.get_t2w_magnitude_any("phy001", "001", 1, allow_multiple_files=True))

    # bids_root = "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids"
    # derivatives = ["/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/derivatives/SIEMENS",
    #                "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/derivatives/FieldMapCorrection"]
    # out_derivatives_folder = "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/derivatives/nipype"
    #
    # dataset = ThreeDimEpiDataset(bids_root, out_derivatives_folder,
    #                                  derivatives)
    # print(dataset.get_file_by_type("t2w_magnitude_raw", subject="phy003", session="001"))