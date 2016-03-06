#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'martin.majer'

import os
import re

import common as common

# set output directory
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, '..', 'data', 'segmented')
output_file = os.path.join(current_dir, '..', 'data', 'reference.txt')

print 'Current directory:\t', current_dir
print 'Data directory:\t\t', data_dir
print 'Output filename:\t', output_file

# get all wav file names
file_names = common.get_files(data_dir, '.wav')[0]

print 'File count:\t', len(file_names)
print 'Sample:\t\t', file_names[:8]

#  generate references
numbers = ['nula', 'jedna ', 'dva', 'tři', 'čtyři', 'pět', 'šest', 'sedum', 'osum', 'devět']

with open(output_file, 'w') as fw:
    fw.write('#!MLF!#\n')

    for file in file_names:
        matchObj = re.match(r'(\d)-(\d)-(\d)\.wav', file)
        index = int(matchObj.group(3))
        file_number = re.sub('\.wav$', '', file)
        fw.write('"*/%s.lab"\n%s\n\n' % (file_number, numbers[index]))
