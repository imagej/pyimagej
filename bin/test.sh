#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

echo "-------------------------------------"
echo "| Testing ImageJ2 + original ImageJ |"
echo "-------------------------------------"
python -m pytest -p no:faulthandler test

echo
echo "-------------------------------------"
echo "|    Testing ImageJ2 standalone     |"
echo "-------------------------------------"
python -m pytest -p no:faulthandler --legacy=false test

echo
echo "-------------------------------------"
echo "|  Testing Fiji Is Just ImageJ(2)   |"
echo "-------------------------------------"
python -m pytest -p no:faulthandler --ij=sc.fiji:fiji test
