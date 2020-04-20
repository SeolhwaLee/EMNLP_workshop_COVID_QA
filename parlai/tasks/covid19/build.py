#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.
import json
import random

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

# RESOURCES = [
#     DownloadableFile(
#         'http://parl.ai/downloads/wikiqa/wikiqa.tar.gz',
#         'wikiqa.tar.gz',
#         '9bb8851dfa8db89a209480e65a3d8967d8bbdf94d5d17a364c0381b0b7609412',
#     )
# ]
RESOURCES = [
    DownloadableFile(
        'https://drive.google.com/file/d/1yPZprx9L1H4G9K7geJup-P0tS_nC0ko3/view?usp=sharing',
        'convid19qa.tar.gz',
        'cfa439496d1d45c53881b90643b1f029',
        True,
        True,
    )
]


def create_fb_format(outpath, dtype, inpath):
    print('building fbformat:' + dtype)
    fout = open(os.path.join(outpath, dtype + '.txt'), 'w')
    with open(inpath) as f:
        lines = [line.strip('\n') for line in f]
    lastqid, lq, ans, cands = None, None, None, None
    for i in range(2, len(lines)):
        l = lines[i].split('\t')
        lqid = l[0]  # question id
        if lqid != lastqid:
            if lastqid is not None:
                # save
                s = '1 ' + lq + '\t' + ans.lstrip('|') + '\t\t' + cands.lstrip('|')
                if (dtype.find('filtered') == -1) or ans != '':
                    fout.write(s + '\n')
            # reset
            cands = ''
            ans = ''
            lastqid = lqid
        lcand = l[5]  # candidate answer / sentence from doc
        lq = l[1]  # question
        llabel = l[6]  # 0 or 1
        if int(llabel) == 1:
            ans = ans + '|' + lcand
        cands = cands + '|' + lcand
    fout.close()

def create_fb_format_covid(outpath, dtype, inpath):
    print('building fbformat:' + dtype)
    fout = open(os.path.join(outpath, dtype + '.txt'), 'w')
    with open(inpath) as f:
        # lines = [line.strip('\n') for line in f]
        data = json.load(f)
    lq, ans, cands = None, None, None
    for line in data:
        # lq = data[i][0]
        lq = str(line[0]).replace('\n', ' ')
        # answer = data[i][1][0] #ground truth
        # answer = line[1][0]
        # cands = answer + '|' + data[i][1][1] + '|' + data[i][1][2]
        # answer = str(line[1][0]).replace('\n', ' ').replace('\r', '')

        cands_pre = [sub.replace('\n', ' ').replace('\r', '').replace('\t', '') for sub in line[1]]
        cands_pre_shuffle = random.sample(cands_pre, len(cands_pre))
        cands = '|'.join(cands_pre_shuffle)
        # answer = cands.split('|')[0]
        answer = cands_pre[0]
        s = '1 ' + lq + '\t' + answer + '\t\t' + cands.lstrip('|')

        fout.write(s + '\n')
    fout.close()



def build(opt):
    dpath = os.path.join(opt['datapath'], 'Covid19QA')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        # for downloadable_file in RESOURCES:
        #     downloadable_file.download_file(dpath)


        dpext = os.path.join(dpath, 'CovidQACorpusScraped')
        create_fb_format_covid(dpath, 'train', os.path.join(dpext, 'Covid19QA-train.json'))
        create_fb_format_covid(dpath, 'valid', os.path.join(dpext, 'Covid19QA-dev.json'))
        create_fb_format_covid(dpath, 'test', os.path.join(dpext, 'Covid19QA-test.json'))
        create_fb_format_covid(
            dpath, 'train-filtered', os.path.join(dpext, 'Covid19QA-train.json')
        )
        create_fb_format_covid(dpath, 'valid-filtered', os.path.join(dpext, 'Covid19QA-dev.json'))
        create_fb_format_covid(dpath, 'test-filtered', os.path.join(dpext, 'Covid19QA-test.json'))

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
