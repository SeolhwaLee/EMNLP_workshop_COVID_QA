#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Interact with a pre-trained model.

This seq2seq model was trained on convai2:self.
"""

from parlai.core.build_data import download_models
from parlai.core.params import ParlaiParser
from parlai.scripts.interactive import interactive, setup_args

if __name__ == '__main__':
    # parser_new = ParlaiParser(add_model_args=True)
    parser = setup_args()
    parser.set_params(
        model='legacy:seq2seq:0',
        model_file='models:convai2/seq2seq/convai2_self_seq2seq_model',
        dict_file='models:convai2/seq2seq/convai2_self_seq2seq_model.dict',
        dict_lower=True,
    )
    parser.add_argument(
        '-sc',
        '--script-chateval',
        type='bool',
        default=False,
        dest='chat_script',
        help='Chateval script read file'
             'True: chateval evaluation, False: single-turn conversation with agent(original model)',
    )
    parser.add_argument(
        '-scip',
        '--chateval-input-path',
        type=str,
        default=None,
        dest='script_input_path',
        help='Chateval script input path',
    )
    parser.add_argument(
        '-scop',
        '--chateval-output-path',
        type=str,
        default=None,
        dest='script_output_path',
        help='Chateval result output path',
    )
    parser.add_argument(
        '--chateval-multi-num',
        type=int,
        default=0,
        dest='chateval_multi_num',
        help='True is chateval multiturn setting, turn coverage count.',
    )
    parser.add_argument(
        '--chateval-multi',
        type='bool',
        default=False,
        hidden=True,
        dest='chateval_multi',
        help='True is chateval multiturn setting, False just single turn.',
    )
    
    opt = parser.parse_args()
    # opt = parser.setup_args()
    if opt.get('model_file', '').startswith('models:convai2'):
        opt['model_type'] = 'seq2seq'
        fnames = [
            'convai2_self_seq2seq_model.tgz',
            'convai2_self_seq2seq_model.dict',
            'convai2_self_seq2seq_model.opt',
        ]
        download_models(opt, fnames, 'convai2', version='v3.0')

    interactive(opt)
