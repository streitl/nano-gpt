#!/usr/bin/env bash
source $(poetry env info --path)/bin/activate

mode=$1

if [[ "$mode" == "--format" ]]; then
  isort_extra=
  black_extra=
else
  if [[ "$mode" == "--style" ]]; then
    isort_extra=--check-only
    black_extra=--check
  else
    echo "unsupported mode: $mode"
    exit 1
  fi
fi

if [ $# -eq 1 ]
then
  files=.
else
  declare -a files_arr
  for file in "${@:2}"; do
      files_arr+=( ${file#"QomonGeoLib/"} )
  done
  files=$(printf " %s" "${files_arr[@]}")
fi

echo "files $files"

printf "black\n"
black $black_extra $files
status_black=$?

printf "\nisort\n"
isort $isort_extra $files
status_isort=$?

printf "\nmypy\n"
mypy $files
status_mypy=$?

printf "\nflake8\n"
flake8 $files
status_flake8=$?

if [ $status_black -eq 0 ] && [ $status_isort -eq 0 ] && [ $status_mypy -eq 0 ] && [ $status_flake8 -eq 0 ]
then
  exit 0
else
  exit 1
fi
