#!/bin/bash
d=0 # Indexes days
i=4 # Indexes pots
j=0 # Indexes plant number within pot
k=0 # Indexes file number
pwd
for FILE in "data/raw_data/exp2_images"/*; do
    if [ $((k%2)) -eq 0 ]; then
        ((j++))
    fi
    if [ $j -eq 9 ]; then
	j=1
	((i++))
    fi
    if [ $i -eq 7 ]; then
	i=4
	((d++))
    fi
    ((k++))
    if [ $((k%2)) -eq 0 ]; then
       cp -r "$FILE" "data/processed_data/exp2_day($d)_pot($i)_plant($j)_top.HEIC"
    fi
    if [ $((k%2)) -eq 1 ]; then
	cp -r "$FILE" "data/processed_data/exp2_day($d)_pot($i)_plant($j)_front.HEIC"
    fi
done
