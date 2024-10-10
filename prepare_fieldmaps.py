from bids.layout import BIDSLayout
from datasets.dzne_three_dim_epi_dataset import DzneThreeDimEpiDataset


if __name__ == "__main__":

    bids_root = "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids"
    derivatives = ["/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/derivatives/SIEMENS",
                   "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/derivatives/FieldMapCorrection"]
    out_derivatives_folder = "/media/laurin/Data_share/Travel_Head_Study/Bonn_Skyra_3T_LowRes_bids/derivatives/nipype"

    dataset = DzneThreeDimEpiDataset(bids_root, out_derivatives_folder, derivatives)

    #
    # layout = BIDSLayout(bids_root, out_derivatives_folder, derivatives=derivatives, validate=False)

    print(dataset.get_b1_anat_ref(subject="phy003", session="001", run=None, generate=True))
    print(dataset.get_b1_in_percent(subject="phy003", session="001", run=None, generate=True))

    # print(dataset.get_t2w_phase_raw("phy001", "001", 1))
    # print(dataset.get_corrected_b1_map("phy001", "001", 1, generate=True))
    # print(dataset.get_t2w_magnitude_any("phy001", "001", 1, allow_multiple_files=True))

