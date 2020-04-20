#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.scripts.interactive import setup_args, interactive
# from parlai.scripts.interactiveDemo import setup_args, interactive
from parlai.scripts.interactiveDemo import setup_args

import random
import argparse
from flask import Flask
from flask_restful import Resource, Api, reqparse

from parlai.core.params import ParlaiParser, get_model_name
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.local_human.local_human import LocalHumanAgent


import logging
import mysql.connector

logging.getLogger("flask_ask").setLevel(logging.DEBUG)

STATUS_OK = "ok"
STATUS_ERROR = "error"

app = Flask(__name__)
api = Api(app)

parser_api = reqparse.RequestParser()

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def db_connect():
    db = mysql.connector.connect(host="35.237.91.101", user="JHU_team", passwd="6Ez1koZgcGbtCA", db="chat_demo")
    return db

db_info = db_connect()

def db_exec(db, arg1, arg2, arg3, arg4):
    c = db.cursor()
    # c.execute('INSERT INTO history ( chat_id, user_utt, sys_utt, model_id, utter_id ) VALUES (%s, %s, %s, %s, %s)',
    #           (int(arg1), str(arg2), str(arg3), int(arg4), int(arg5)))
    c.execute('INSERT INTO history ( chat_id, user_utt, sys_utt, model_id) VALUES (%s, %s, %s, %s)',
              (int(arg1), str(arg2), str(arg3), int(arg4)))
    db.commit()
    c.close()

def db_close(db):
    return db.close()

# def interactive(args, opt, print_parser=None):
#     if print_parser is not None:
#         if print_parser is True and isinstance(opt, ParlaiParser):
#             print_parser = opt
#         elif print_parser is False:
#             print_parser = None
#     if isinstance(opt, ParlaiParser):
#         print('[ Deprecated Warning: interactive should be passed opt not Parser ]')
#         opt = opt.parse_args()
#
#     print("interact check 1 !!!!!!!")
#     print("args check!!!!!!!!!!!!!", args)
#     # Create model and assign it to the spepython projects/controllable_dialogue/interactiveDemo.py -mf models:controllable_dialogue/control_avgnidf10b10e -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20 --set-controls avg_nidf:7cified task
#     agent = create_agent(opt, requireModelExists=True)
#
#     print("interact check 2 !!!!!!!")
#     human_agent = LocalHumanAgent(opt)
#     print("interact check 3 !!!!!!!")
#     world = create_task(opt, [human_agent, agent])
#     print("interact check 4 !!!!!!!")
#     if print_parser:
#         # Show arguments after loading model
#         print_parser.opt = agent.opt
#         print_parser.print_args()
#
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     # Show some example dialogs:
#     if not opt.get('chat_script'):
#         '''for chateval script evaluation'''
#         while True:
#             # for conversation in terminal
#             # world.parley()
#             # for demo
#             # print("!!", args)
#             # args = 'hello'
#             bot_message = world.parley_demo(args)
#             # print("bot", bot_message)
#             # world.parley_demo(str(args))
#             if opt.get('display_examples'):
#                 print("---")
#                 print(world.display())
#             if world.epoch_done():
#                 print("EPOCH DONE")
#                 break
#             return bot_message
#     else:
#         model = get_model_name(opt)
#         while True:
#             world.parley_script(opt.get('script_input_path'), opt.get('script_output_path'), str(model))
#             if opt.get('display_examples'):
#                 print("---")
#                 print(world.display())
#             if world.epoch_done():
#                 print("EPOCH DONE")
#                 break


class Service(Resource):

    def NLG(self, input):

        # insert your NLG solution here
        # output = "you said " + input
        # inputs = [{"id": 0, "src": input}]

        try:
            # Run chatbot with GPT-2
            # bot_message, utt_id_ = run_chat(model, tokenizer, config, input)
            # bot_message = run_chat(model, tokenizer, config, input)
            random.seed(42)
            parser = setup_args()

            print("!!", str(input))

            # def interactive(args, opt, print_parser=None):
            # bot_message = interactive(str(input), parser.parse_args(print_args=False), print_parser=parser)
            bot_message = interactive(parser.parse_args(print_args=False), print_parser=parser)
            # interactive(parser.parse_args(print_args=False), print_parser=parser)
            # interactive(parser.parse_args(print_args=False), print_parser=parser)

            print("bot", bot_message)

            output = bot_message
            # print(output)

            if str(input).strip().lower() == '/bye':
                print("{ inputs:", input, " response:", output, "}")
                # db_exec(db_info, '1', '/bye:reset', '/bye:reset', '1', '1')
                db_exec(db_info, '1', '/bye:reset', '/bye:reset', '3')

            elif str(input).strip().lower() == '/stop':
                print("{ inputs:", input, " response:", output, "}")
                # db_exec(db_info, '1', '/bye:reset', '/bye:reset', '1', '1')
                db_exec(db_info, '1', '/stop:done', '/stop:done', '3')

            elif str(input).strip().lower() == '/start':
                print("{ inputs:", input, " response:", output, "}")
                # db_exec(db_info, '1', '/bye:reset', '/bye:reset', '1', '1')
                db_exec(db_info, '1', '/start', '/start', '3')

            else:
                print("{ inputs:", input, " response:", output, "}")
                user_utt, sys_utt = input, output
                # db_exec(db_info, '1', user_utt, sys_utt, '1', '1')
                db_exec(db_info, '1', user_utt, sys_utt, '1')


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

    parser.set_params(batchsize=1, beam_size=20, beam_min_n_best=10)

    print('\n' + '*' * 80)
    print('WARNING: This dialogue model is a research project that was trained on a')
    print(
        'large amount of open-domain Twitter data. It may generate offensive content.'
    )
    print('*' * 80 + '\n')

    # interactive(parser.parse_args(print_args=False), print_parser=parser)
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)
    db_close(db_info)