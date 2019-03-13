#!/bin/bash

str=$(strings samples/benignware/d9fcf8051179f9a6ed0a1ee42c62320ea3fde31d | tr '\n' '=' | sed 's/=/","/g')
echo -n "[\""
echo -n "${str::-2}"
echo "]"
