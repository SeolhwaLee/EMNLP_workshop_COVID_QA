#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic example which allows local human keyboard input to talk to a trained model.

For documentation, see parlai.scripts.interactive.
"""
from parlai.scripts.interactiveCovid19 import setup_args, interactive
import random
# from parlai.scripts.interactiveDemo import setup_args

import random
from flask import Flask
from flask_restful import Resource, Api, reqparse
import logging

logging.getLogger("flask_ask").setLevel(logging.DEBUG)

STATUS_OK = "ok"
STATUS_ERROR = "error"

app = Flask(__name__)
api = Api(app)

parser_api = reqparse.RequestParser()

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class Service(Resource):

    def NLG(self, input_text):

        # insert your NLG solution here
        # output = "you said " + input
        # inputs = [{"id": 0, "src": input}]

        try:
            # Run chatbot with GPT-2
            # bot_message, utt_id_ = run_chat(model, tokenizer, config, input)
            # bot_message = run_chat(model, tokenizer, config, input)
            random.seed(42)
            parser = setup_args()

            print("!!", str(input_text))

            # def interactive(args, opt, print_parser=None):
            # bot_message = interactive(str(input), parser.parse_args(print_args=False), print_parser=parser)
            opt = parser.parse_args()
            bot_message = interactive(opt, print_parser=parser, raw_text=input_text)
            # interactive(parser.parse_args(print_args=False), print_parser=parser)
            # interactive(parser.parse_args(print_args=False), print_parser=parser)

            print("bot", bot_message)

            output = bot_message
            # print(output)

        except:
            pass

        return(output)

    def post(self):
        parser_api.add_argument('input', type=str)
        args_api = parser_api.parse_args()

        output = self.NLG(args_api['input'])
        return {
            'output': '{}'.format(output)
        }


api.add_resource(Service, '/')

app.config['PROPAGATE_EXCEPTIONS'] = True

if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    # opt = parser.parse_args()
    # interactive(opt, print_parser=parser)

    print('\n' + '*' * 80)
    print('WARNING: This dialogue model is a research project that was trained on a')
    print(
        'large amount of open-domain Twitter data. It may generate offensive content.'
    )
    print('*' * 80 + '\n')

    # interactive(parser.parse_args(print_args=False), print_parser=parser)
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)

