#!/bin/sh

dir=$(dirname "$0")
cd "$dir"

echo "-------------------------------------"
echo "| Testing ImageJ2 + original ImageJ |"
echo "-------------------------------------"
pytest -p no:faulthandler

echo
echo "-------------------------------------"
echo "|    Testing ImageJ2 standalone     |"
echo "-------------------------------------"
pytest -p no:faulthandler --legacy=false

echo
echo "-------------------------------------"
echo "|  Testing Fiji Is Just ImageJ(2)   |"
echo "-------------------------------------"
pytest -p no:faulthandler --ij=sc.fiji:fiji
