#!/bin/bash

array=('abc'
       'cde'
       3
       4
       5)
printf ${array[1]}
printf ${array[2]}
#for i in (1..$array[@])
#do
#    printf $array[$i]
#done
