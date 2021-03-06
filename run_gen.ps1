Write-Host "Generate log file";
python genLog.py --p=$args
Write-Host "Transcribing basic actions";
python preprocessing_sniffer_log.py --p=$args
python transcribe_basicaction.py --p=$args
Write-Host "Training the detectors";
python gen_script_detectors.py --p=$args