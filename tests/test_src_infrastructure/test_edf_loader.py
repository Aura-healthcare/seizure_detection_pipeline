from src.infrastructure.edf_loader import EdfLoader

TEST_TUH_EDF_FILENAME = \
    'data/tuh/dev/01_tcp_ar/002/00009578/00009578_s002_t001.edf'


def test_get_ecg_candidate_channel_no_element():

    loader = EdfLoader(edf_file_path=TEST_TUH_EDF_FILENAME)
    loader.channels = []
    ecg_candididate_channel = loader.get_ecg_candidate_channel()
    assert(ecg_candididate_channel is None)
