#!/bin/bash
# Ellis Brown

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    # navigate to ~/data
    echo "navigating to ~/data/ ..." 
    mkdir -p ~/data
    cd ~/data/
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "navigating to" $1 "..."
    cd $1
fi

echo "Downloading VOC2012 trainval ..."
# Download the data.
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
echo "Done downloading."

# Extract data
echo "Extracting trainval ..."
tar -xvf VOCtrainval_11-May-2012.tar
echo "removing tar ..."
rm VOCtrainval_11-May-2012.tar

# Add ImageSets
echo "navigating to ./VOCdevkit/VOC2012/ImageSets/ ..."
cd ./VOCdevkit/VOC2012/ImageSets/
echo "Downloading VOC2012 ImageSets ..."
wget https://github.com/Ze-Yang/ImageSets/raw/master/Main2012.tar
echo "Extracting VOC2012 ImageSets ..."
tar -xvf Main2012.tar
echo "removing tars ..."
rm Main2012.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"