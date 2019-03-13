#!/bin/bash

if [[ $# -lt 1 ]]; then
	exit 1
fi

str=$(strings samples/$1 | tr '\n' '=' | sed 's/=/@/g')
echo -n "\""
echo -n "${str::-2}"
#echo -n "]"
