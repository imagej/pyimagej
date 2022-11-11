#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

modes="
| Testing ImageJ2 + original ImageJ |--legacy=true
|    Testing ImageJ2 standalone     |--legacy=false
|  Testing Fiji Is Just ImageJ(2)   |--ij=sc.fiji:fiji
"

echo "$modes" | while read mode
do
  test "$mode" || continue
  msg="${mode%|*}|"
  flag=${mode##*|}
  echo "-------------------------------------"
  echo "$msg"
  echo "-------------------------------------"
  if [ $# -gt 0 ]
  then
    python -m pytest -p no:faulthandler $flag $@
  else
    python -m pytest -p no:faulthandler $flag tests
  fi
done
