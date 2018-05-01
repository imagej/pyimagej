#!/bin/bash

sudo apt-get update

# -- create a test enviroment --
conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION
source activate test-environment

# -- install dependencies --
pip install Cython
pip install pyjnius

# -- install supporting tools --
sudo apt -y install curl
sudo apt -y install git
sudo apt -y install unzip

cd $HOME

# -- download Fiji.app --
if [ ! -d Fiji.app ]
then
  echo
  echo "--> Downloading Fiji"
  curl -fsO http://downloads.imagej.net/fiji/latest/fiji-nojre.zip

  echo "--> Unpacking Fiji"
  rm -rf Fiji.app
  unzip fiji-nojre.zip
fi

# -- Determine correct ImageJ launcher executable --
case "$(uname -s),$(uname -m)" in
  Linux,x86_64) launcher=ImageJ-linux64 ;;
  Linux,*) launcher=ImageJ-linux32 ;;
  Darwin,*) launcher=Contents/MacOS/ImageJ-macosx ;;
  MING*,*) launcher=ImageJ-win32.exe ;;
  *) die "Unknown platform" ;;
esac

echo
echo "--> Updating Fiji"
Fiji.app/$launcher --update update-force-pristine

# -- install imglyb --
conda install -c hanslovsky imglib2-imglyb

# -- clone testing unit --
git clone https://github.com/imagej/imagej.py.git
cd $HOME/imagej.py
git checkout pyjnius

# -- set ij dirctory --
ij_dir=$HOME/Fiji.app
echo $ij_dir
python setup.py install

# -- run test with debug flag --
cd test
python -O test_imagej.py --ij $ij_dir

