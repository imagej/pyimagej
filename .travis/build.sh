#!/bin/bash

set -e

die () {
	echo "$*" >&2
	exit 1
}

check () {
	while test $# -gt 0
	do
		which "$1" || sudo apt -y install "$1" || die "Could not install $1"
		shift
	done
}

# -- create a test enviroment --
conda create -n imagej -y python=$TRAVIS_PYTHON_VERSION
source activate imagej
conda env update -f environment.yml
conda install pytest

# -- ensure supporting tools are available --
check curl git unzip

cd

# -- download Fiji.app --
if [ ! -d Fiji.app ]
then
	echo
	echo "--> Downloading Fiji"
	curl -fsO https://downloads.imagej.net/fiji/latest/fiji-nojre.zip

	echo "--> Unpacking Fiji"
	rm -rf Fiji.app
	unzip fiji-nojre.zip
fi

# -- Determine correct ImageJ launcher executable --
case "$(uname -s),$(uname -m)" in
	Linux,x86_64) launcher=ImageJ-linux64 ;;
	Linux,*) launcher=ImageJ-linux32 ;;
	Darwin,*) launcher=Contents/MacOS/ImageJ-macosx; skipGUI=1 ;;
	MING*,*) launcher=ImageJ-win32.exe ;;
	*) die "Unknown platform" ;;
esac

echo
echo "--> Updating Fiji"
Fiji.app/$launcher --update update-force-pristine

# -- run the Python code --
cd $TRAVIS_BUILD_DIR

# -- set ij dirctory --
ij_dir=$HOME/Fiji.app
echo "ij_dir = $ij_dir"
python setup.py install

# -- unset JAVA_HOME in case it was set --
unset JAVA_HOME

# -- run tests with local Fiji.app --
python -m pytest --ij="$ij_dir"
test "$skipGUI" || python -m pytest --ij="$ij_dir" --headless=false

# -- run tests with ImageJ from Maven repository --
python -m pytest

