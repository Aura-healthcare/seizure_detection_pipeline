FOLDER_PATH= .
SRC_PATH=./src
TEST_PATH=./tests


clean:
	rm output/db/*csv
	
example_detect_qrs:
	python3 src/usecase/detect_qrs.py --filename data/tuh/dev/01_tcp_ar/002/00009578/00009578_s006_t001.edf --method hamilton --exam-id 00009578_s006_t001
	
example_ecg_qc:
	python3 src/usecase/apply_ecg_qc.py --filename data/tuh/dev/01_tcp_ar/002/00009578/00009578_s006_t001.edf --sampling-frequency 1000 --model rfc_2s.pkl --exam-id 00009578_s006_t001

example_hrv:
	python3 src/usecase/compute_hrvanalysis_features.py --rr-intervals-file-path output/rr_intervals/00009578_s006_t001.csv --output-folder .


test:
	pytest -s -vvv $(TEST_PATH)

coverage:
	pytest --cov=$(SRC_PATH) --cov-report html $(TEST_PATH) 