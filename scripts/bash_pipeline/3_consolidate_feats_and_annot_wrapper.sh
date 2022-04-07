#!/bin/bash

## Option - select input directory to be copied from and output directory to copy into
while getopts ":i:o:a:" option
do
case "${option}"
in
i) InputDest=${OPTARG};;
a) AnnotDest=${OPTARG};;
o) TargetDest=${OPTARG};;
esac
done

ECG_PATH=$(pwd)

# Check script input integrity
if [[ $InputDest ]] || [[ $TargetDest ]];
then
  echo "Start Executing script"
else
  echo "No Input feats directory: $InputDest or no input Annot directory: $AnnotDest or Target directory: $TargetDest, use -i,-a,-o options" >&2
  exit 1
fi


## Copy TUH folder tree structure
TargetDest=$(realpath "$TargetDest")
mkdir -p "$TargetDest";
cd "$InputDest" && find . -type d -exec mkdir -p -- "$TargetDest/{}" \; && cd -

#Hack to force the bash to split command result only on newline char
#It is done to support the spaces in the folder names
OIFS="$IFS"
IFS=$'\n'

## List all rr_files in InputDest ##
for features_file in $(find "$InputDest"/* -type f -name "*.csv" ); do

    filename=$(echo "$features_file" | awk -F/ '{print $NF}')
    # Get relative path
    path=$(echo "$features_file" | sed "s/$filename//g")
    CleanDest=$(echo "$InputDest" | sed 's/\//\\\//g')
    relative_path=$(echo "$path" | sed "s/$CleanDest\///g")

    if [ "$(basename "$path" | head -c 4)" == "PAT_" ];
    then # Dataset format
      tse_bi_filename=${relative_path::-1}_Annotations_${filename:6:-7}.tse_bi
    else # TUH format
      tse_bi_filename=$(cut -c7- <<< "${filename%.*}.tse_bi")
    fi

    annotation_file_path=$AnnotDest/$relative_path$tse_bi_filename

	  python3 "$ECG_PATH/src/usecase/consolidate_feats_and_annot.py" --features-file-path "$features_file" --annotations-file-path "$annotation_file_path" --output-folder "$TargetDest/$relative_path"

    if [ $? -eq 0 ]
    then
      echo "$features_file and $annotation_file_path # - OK"
    else
      echo "$features_file and $annotation_file_path # - Fail" >&2
    fi

done

#Restore the bash automatic split on spaces
IFS="$OIFS"

exit 0
