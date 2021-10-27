import argparse
from src.usecase.detect_qrs import detect_qrs, parse_detect_qrs_args

TEST_TUH_EDF_FILENAME = \
    'data/tuh/dev/01_tcp_ar/002/00009578/00009578_s002_t001.edf'
TEST_TUH_EXAM_ID = '1234537'

OUTPUT_FOLDER = 'tests/output'
TEST_METHODS = ['hamilton', 'xqrs', 'gqrs', 'swt', 'engelsee']


def test_tuh_detect_qrs():

    for method in TEST_METHODS:
        sampling_frequency, filename = detect_qrs(
            filename=TEST_TUH_EDF_FILENAME,
            method=method,
            exam_id=TEST_TUH_EXAM_ID,
            output_folder=OUTPUT_FOLDER)

        assert(sampling_frequency == 250)
        assert (filename == ''.join([OUTPUT_FOLDER,
                                     '/',
                                     TEST_TUH_EXAM_ID,
                                     '.csv']))


def test_tuh_parse_detect_qrs_args():

    bash_command = (f'python3 src/usecase/detect_qrs.py '
                    f'--filename {TEST_TUH_EDF_FILENAME} '
                    f'--method hamilton '
                    f'--exam-id {TEST_TUH_EXAM_ID} '
                    f'--output-folder {OUTPUT_FOLDER}')
    args_to_parse = bash_command.split()[2:]
    parser = parse_detect_qrs_args(args_to_parse)

    correct_parser = argparse.Namespace(
        exam_id=TEST_TUH_EXAM_ID,
        filename=TEST_TUH_EDF_FILENAME,
        method='hamilton',
        output_folder=OUTPUT_FOLDER)
    assert(parser == correct_parser)
