#!/bin/sh

dir=$(dirname "$0")
cd "$dir/.."

modes="
| Testing ImageJ2 + original ImageJ |--legacy=true
|    Testing ImageJ2 standalone     |--legacy=false
|  Testing Fiji Is Just ImageJ(2)   |--ij=sc.fiji:fiji
|  Testing local Fiji-Stable        |--ij=Fiji.app
|  Testing local Fiji-Latest        |--ij=Fiji
|  Testing ImageJ2 version 2.10.0   |--ij=2.10.0
|  Testing ImageJ2 version 2.14.0   |--ij=2.14.0
"

if [ ! -d Fiji.app ]
then
  # No locally available Fiji-Stable; download one.
  echo "-- Downloading and unpacking Fiji-Stable --"
  curl -fsLO https://downloads.imagej.net/fiji/stable/fiji-stable-portable-nojava.zip
  unzip fiji-stable-portable-nojava.zip
  echo
fi

if [ ! -d Fiji ]
then
  # No locally available Fiji-Latest; download one.
  echo "-- Downloading and unpacking Fiji-Latest --"
  curl -fsLO https://downloads.imagej.net/fiji/latest/fiji-latest-portable-nojava.zip
  unzip fiji-latest-portable-nojava.zip
  echo
fi

echo "$modes" | while read mode
do
  test "$mode" || continue
  msg="${mode%|*}|"
  flag=${mode##*|}
  for java in 11 21
  do
    # Fiji-Latest requires Java 21; skip Fiji-Latest with older Javas.
    echo "$msg" | grep -q Fiji-Latest && test "$java" -lt 21 && continue

    echo "-------------------------------------"
    echo "$msg"
    printf "|           < OpenJDK %2s >          |\n" "$java"
    echo "-------------------------------------"
    if [ $# -gt 0 ]
    then
      uv run python -m pytest -p no:faulthandler $flag --java $java $@
    else
      uv run python -m pytest -p no:faulthandler $flag --java $java tests
    fi
    code=$?
    if [ $code -eq 0 ]
    then
      echo "==> TESTS PASSED with code 0"
    else
      # HACK: `while read` creates a subshell, which can't modify the parent
      # shell's variables. So we save the failure code to a temporary file.
      echo
      echo "==> TESTS FAILED with code $code"
      echo $code >exitCode.tmp
    fi
    echo
  done
done
exitCode=0
if [ -f exitCode.tmp ]
then
  exitCode=$(cat exitCode.tmp)
  rm -f exitCode.tmp
fi
exit "$exitCode"
