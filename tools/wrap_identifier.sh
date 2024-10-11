#!/usr/bin/env zsh
# author: Cauphenuny

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 filein list version fileout"
    exit 1
fi

filein="$1"
list="$2"
version="$3"
fileout="$4"

echo $list | read -A list

if [ ! -f "$filein" ]; then
    echo "Input file $filein not found!"
    exit 1
fi

while IFS= read -r line
do
    for id in ${list[@]}
    do 
        line=${line//\<$id\>/$id$version}
    done
    echo $line
done < $filein > $fileout
