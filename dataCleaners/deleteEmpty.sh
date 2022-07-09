#!/bin/bash

x=$(sed -n '$=' $1)
xargs -a $1 -I{} -d'\n' find ../train/ -maxdepth 2 -type f -name {} -delete
echo "$x files deleted"
