#! /bin/bash

# This file I copied from StackOverflow and slightly tweaked it.
# It splits the provided file into multiple files and appends "part#" to
# the file name.

fileDir="$1"
file="$2"
fileLength=$(wc -l < "$fileDir/$file")
echo $fileDir
echo $file
echo $fileLength
shift

part=1
percentSum=0
currentLine=1
echo "$@"
for percent in "${@:2}"; do
        filename=${file%.*}
        ext=${file##*.}
        [ "$percent" == "." ] && ((percent = 100 - percentSum))
        filePart="$filename-part$part.$ext"
        echo $filePart
        ((percentSum += percent))
        if ((percent < 0 || percentSum > 100)); then
                echo "invalid percentage" 1>&2
                exit 1
        fi
        ((nextLine = fileLength * percentSum / 100))
        if ((nextLine < currentLine)); then
                printf "" # create empty file
        else
                sed -n "$currentLine,$nextLine"p "$fileDir/$file"
        fi > "$fileDir/$filePart"
        ((currentLine = nextLine + 1))
        ((part++))
done
exit 0