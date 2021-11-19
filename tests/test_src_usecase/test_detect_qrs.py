import argparse
from src.usecase.detect_qrs import detect_qrs, parse_detect_qrs_args,\
    parse_exam_id
from src.usecase.utilities import convert_args_to_dict
TEST_TUH_EDF_FILENAME = \
    'data/tuh/dev/01_tcp_ar/002/00009578/00009578_s002_t001.edf'
TEST_TUH_EXAM_ID = '1234536'

OUTPUT_FOLDER = 'tests/output'
TEST_METHODS = ['hamilton', 'xqrs', 'gqrs', 'swt', 'engelsee']


def test_tuh_detect_qrs():

    for method in TEST_METHODS:
        export_filename, sampling_frequency = detect_qrs(
            qrs_file_path=TEST_TUH_EDF_FILENAME,
            method=method,
            exam_id=TEST_TUH_EXAM_ID,
            output_folder=OUTPUT_FOLDER)

        assert(sampling_frequency == 250)
        assert (export_filename == ''.join(
            [OUTPUT_FOLDER,
             '/',
             'rr_',
             TEST_TUH_EXAM_ID,
             '.csv']))


def test_tuh_parse_detect_qrs_args():

    bash_command = (f'python3 src/usecase/detect_qrs.py '
                    f'--qrs-file-path {TEST_TUH_EDF_FILENAME} '
                    f'--method hamilton '
                    f'--exam-id {TEST_TUH_EXAM_ID} '
                    f'--output-folder {OUTPUT_FOLDER}')
    args_to_parse = bash_command.split()[2:]
    parser = parse_detect_qrs_args(args_to_parse)
    parser_dict = convert_args_to_dict(parser)

    correct_parser = argparse.Namespace(
        qrs_file_path=TEST_TUH_EDF_FILENAME,
        method='hamilton',
        exam_id=TEST_TUH_EXAM_ID,
        output_folder=OUTPUT_FOLDER)
    correct_parser_dict = convert_args_to_dict(correct_parser)

    assert(parser_dict == correct_parser_dict)


def test_parse_exam_id():
    exam_id = parse_exam_id(qrs_file_path=TEST_TUH_EDF_FILENAME)
    assert(exam_id == '00009578_s002_t001.edf')
