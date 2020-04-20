#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic script which allows local human keyboard input to talk to a trained model.

Examples
--------

.. code-block:: shell

  python examples/interactive.py -m drqa -mf "models:drqa/squad/model"

When prompted, enter something like: ``Bob is Blue.\\nWhat is Bob?``

Input is often model or task specific, but in drqa, it is always
``context '\\n' question``.
"""
import copy
import os

from parlai.core.params import ParlaiParser, get_model_name
# from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.core.loader import load_agent_module
from parlai.utils.misc import warn_once
from parlai.core.opt import Opt, load_opt_file
from parlai.core.build_data import modelzoo_path

import random

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Interactive chat with a model')
    parser.add_argument('-d', '--display-examples', type='bool', default=False)

    parser.add_argument(
        '--display-prettify',
        type='bool',
        default=False,
        help='Set to use a prettytable when displaying '
        'examples with text candidates',
    )
    parser.add_argument(
        '--display-ignore-fields',
        type=str,
        default='label_candidates,text_candidates',
        help='Do not display these fields',
    )
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
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

    parser.set_defaults(interactive_mode=True, task='interactive')
    LocalHumanAgent.add_cmdline_args(parser)
    return parser

class Agent(object):
    """
    Base class for all other agents.
    """

    def __init__(self, opt: Opt, shared=None):
        if not hasattr(self, 'id'):
            self.id = 'agent'
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        self.observation = None

    def observe(self, observation):
        """
        Receive an observation/action dict.
        """
        self.observation = observation
        return observation

    def act(self):
        """
        Return an observation/action dict based upon given observation.
        """
        if hasattr(self, 'observation') and self.observation is not None:
            print('agent received observation:')
            print(self.observation)

        t = {}
        t['text'] = 'hello, teacher!'
        print('agent sending message:')
        print(t)
        return t

    def getID(self):
        """
        Return the agent ID.
        """
        return self.id

    def epoch_done(self):
        """
        Return whether the epoch is done or not.

        :rtype: boolean
        """
        return False

    def reset(self):
        """
        Reset the agent, clearing its observation.

        Many subclasses implement additional reset logic.
        """
        self.observation = None

    def reset_metrics(self):
        """
        Reset any metrics reported by this agent.

        This is called to indicate metrics should start fresh, and is typically called
        between loggings or after a `report()`.
        """
        pass

    def save(self, path=None):
        """
        Save any parameters needed to recreate this agent from loaded parameters.

        Default implementation is no-op, but many subagents implement this logic.
        """
        pass

    def clone(self):
        """
        Make a shared copy of this agent.

        Should be the same as using create_agent_from_shared(.), but slightly easier.
        """
        return type(self)(self.opt, self.share())

    def share(self):
        """
        Share any parameters needed to create a shared version of this agent.

        Default implementation shares the class and the opt, but most agents will want
        to also add model weights, teacher data, etc. This especially useful for
        avoiding providing pointers to large objects to all agents in a batch.
        """
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        return shared

    def shutdown(self):
        """
        Perform any final cleanup if needed.
        """
        pass

    @classmethod
    def upgrade_opt(cls, opt_from_disk: Opt):
        """
        Upgrade legacy options when loading an opt file from disk.

        This is primarily made available to provide a safe space to handle
        backwards-compatible behavior. For example, perhaps we introduce a
        new option today, which wasn't previously available. We can have the
        argument have a new default, but fall back to the "legacy" compatibility
        behavior if the option doesn't exist.

        ``upgrade_opt`` provides an opportunity for such checks for backwards
        compatibility. It is called shortly after loading the opt file from
        disk, and is called before the Agent is initialized.

        Other possible examples include:

            1. Renaming an option,
            2. Deprecating an old option,
            3. Splitting coupled behavior, etc.

        Implementations of ``upgrade_opt`` should conform to high standards,
        due to the risk of these methods becoming complicated and difficult to
        reason about. We recommend the following behaviors:

            1. ``upgrade_opt`` should only be used to provide backwards
            compatibility.  Other behavior should find a different location.
            2. Children should always call the parent's ``upgrade_opt`` first.
            3. ``upgrade_opt`` should always warn when an option was overwritten.
            4. Include comments annotating the date and purpose of each upgrade.
            5. Add an integration test which ensures your old work behaves
            appropriately.

        :param Opt opt_from_disk:
            The opt file, as loaded from the ``.opt`` file on disk.
        :return:
            The modified options
        :rtype:
            Opt
        """
        # 2019-07-11: currently a no-op.
        return opt_from_disk

def compare_init_model_opts(opt: Opt, curr_opt: Opt):
    """
    Print loud warning when `init_model` opts differ from previous configuration.
    """
    if opt.get('init_model') is None:
        return
    opt['init_model'] = modelzoo_path(opt['datapath'], opt['init_model'])
    optfile = opt['init_model'] + '.opt'
    if not os.path.isfile(optfile):
        return
    init_model_opt = load_opt_file(optfile)

    extra_opts = {}
    different_opts = {}
    exempt_opts = [
        'model_file',
        'dict_file',
        'override',
        'starttime',
        'init_model',
        'batchindex',
    ]

    # search through init model opts
    for k, v in init_model_opt.items():
        if (
            k not in exempt_opts
            and k in init_model_opt
            and init_model_opt[k] != curr_opt.get(k)
        ):
            if isinstance(v, list):
                if init_model_opt[k] != list(curr_opt[k]):
                    different_opts[k] = ','.join([str(x) for x in v])
            else:
                different_opts[k] = v

    # search through opts to load
    for k, v in curr_opt.items():
        if k not in exempt_opts and k not in init_model_opt:
            if isinstance(v, list):
                extra_opts[k] = ','.join([str(x) for x in v])
            else:
                extra_opts[k] = v

    # print warnings
    extra_strs = ['{}: {}'.format(k, v) for k, v in extra_opts.items()]
    if extra_strs:
        print('\n' + '*' * 75)
        print(
            '[ WARNING ] : your model is being loaded with opts that do not '
            'exist in the model you are initializing the weights with: '
            '{}'.format(','.join(extra_strs))
        )

    different_strs = [
        '--{} {}'.format(k, v).replace('_', '-') for k, v in different_opts.items()
    ]
    if different_strs:
        print('\n' + '*' * 75)
        print(
            '[ WARNING ] : your model is being loaded with opts that differ '
            'from the model you are initializing the weights with. Add the '
            'following args to your run command to change this: \n'
            '\n{}'.format(' '.join(different_strs))
        )
        print('*' * 75)

def create_agent_from_opt_file(opt: Opt):
    """
    Load agent options and module from file if opt file exists.

    Checks to see if file exists opt['model_file'] + ".opt"; if so, load up the
    options from the file and use that to create an agent, loading the model
    type from that file and overriding any options specified in that file when
    instantiating the agent.

    If that file does not exist, return None.
    """
    model_file = opt['model_file']
    optfile = model_file + '.opt'
    print("create_agent_from_opt_file!!!!!!!!!!!!!!!!!!!!!!!!!")
    if os.path.isfile(optfile):
        new_opt = load_opt_file(optfile)
        print("create_agent_from_opt_file----222222!!!!!!!!!!!!!!!!!!!!!!!!!")
        # TODO we need a better way to say these options are never copied...
        if 'datapath' in new_opt:
            # never use the datapath from an opt dump
            del new_opt['datapath']
        if 'batchindex' in new_opt:
            # This saved variable can cause trouble if we switch to BS=1 at test time
            del new_opt['batchindex']
        # only override opts specified in 'override' dict
        if opt.get('override'):
            for k, v in opt['override'].items():
                if str(v) != str(new_opt.get(k, None)):
                    print(
                        "[ warning: overriding opt['{}'] to {} ("
                        "previously: {} )]".format(k, v, new_opt.get(k, None))
                    )
                new_opt[k] = v

        print("create_agent_from_opt_file----3333333!!!!!!!!!!!!!!!!!!!!!!!!!")
        model_class = load_agent_module(new_opt['model'])

        print("new check-------------------------", model_class)
        print("create_agent_from_opt_file----4444444!!!!!!!!!!!!!!!!!!!!!!!!!")
        # check for model version
        if hasattr(model_class, 'model_version'):
            print("==================model versio==========")
            curr_version = new_opt.get('model_version', 0)
            if curr_version != model_class.model_version():
                model = new_opt['model']
                print("==================model versio2222222==========")
                m = (
                    'It looks like you are trying to load an older version of'
                    ' the selected model. Change your model argument to use '
                    'the old version from parlai/agents/legacy_agents: for '
                    'example: `-m legacy:{m}:{v}` or '
                    '`--model parlai.agents.legacy_agents.{m}.{m}_v{v}:{c}`'
                )
                if '.' not in model:
                    # give specific error message if it's easy
                    raise RuntimeError(
                        m.format(m=model, v=curr_version, c=model_class.__name__)
                    )
                else:
                    # otherwise generic one
                    raise RuntimeError(
                        m.format(m='modelname', v=curr_version, c='ModelAgent')
                    )

        if hasattr(model_class, 'upgrade_opt'):
            print("==================upgrade versio==========")
            new_opt = model_class.upgrade_opt(new_opt)
            print("==================new_opt versio==========", new_opt)

        # add model arguments to new_opt if they aren't in new_opt already
        for k, v in opt.items():
            if k not in new_opt:
                new_opt[k] = v
        new_opt['model_file'] = model_file
        print("==================new_opt model_file==========", new_opt['model_file'])
        if not new_opt.get('dict_file'):
            new_opt['dict_file'] = model_file + '.dict'
        elif new_opt.get('dict_file') and not os.path.isfile(new_opt['dict_file']):
            old_dict_file = new_opt['dict_file']
            new_opt['dict_file'] = model_file + '.dict'
        if not os.path.isfile(new_opt['dict_file']):
            warn_once(
                'WARNING: Neither the specified dict file ({}) nor the '
                '`model_file`.dict file ({}) exists, check to make sure either '
                'is correct. This may manifest as a shape mismatch later '
                'on.'.format(old_dict_file, new_opt['dict_file'])
            )

        # if we want to load weights from --init-model, compare opts with
        # loaded ones
        compare_init_model_opts(opt, new_opt)
        print("create_agent_from_opt_file----final!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("checkckcckck-----", model_class)
        print("checkckcckck2222222-----", model_class(new_opt))
        return model_class(new_opt)
    else:
        return None

def create_agent(opt: Opt, requireModelExists=False):
    """
    Create an agent from the options ``model``, ``model_params`` and ``model_file``.

    The input is either of the form
    ``parlai.agents.ir_baseline.agents:IrBaselineAgent`` (i.e. the path
    followed by the class name) or else just ``ir_baseline`` which
    assumes the path above, and a class name suffixed with 'Agent'.

    If ``model-file`` is available in the options this function can also
    attempt to load the model from that location instead. This avoids having to
    specify all the other options necessary to set up the model including its
    name as they are all loaded from the options file if it exists (the file
    opt['model_file'] + '.opt' must exist and contain a pickled or json dict
    containing the model's options).
    """
    print("create_agent()111111")
    if opt.get('datapath', None) is None:
        # add datapath, it is missing
        print("create_agent()222222")
        from parlai.core.params import ParlaiParser, get_model_name

        parser = ParlaiParser(add_parlai_args=False)
        parser.add_parlai_data_path()
        # add model args if they are missing
        model = get_model_name(opt)
        if model is not None:
            parser.add_model_subargs(model)
        opt_parser = parser.parse_args("", print_args=False)
        for k, v in opt_parser.items():
            if k not in opt:
                opt[k] = v

    if opt.get('model_file'):
        print("create_agent()3333333")
        opt['model_file'] = modelzoo_path(opt.get('datapath'), opt['model_file'])
        print("eee", opt['model_file'])
        print("create_agent()444444")
        if requireModelExists and not os.path.isfile(opt['model_file']):
            raise RuntimeError(
                'WARNING: Model file does not exist, check to make '
                'sure it is correct: {}'.format(opt['model_file'])
            )
        # Attempt to load the model from the model file first (this way we do
        # not even have to specify the model name as a parameter)
        print("create_agent()555555")
        model = create_agent_from_opt_file(opt)
        # model = load_agent_module(opt)
        print("create_agent()666666")
        if model is not None:
            return model
        else:
            print(f"[ no model with opt yet at: {opt['model_file']}(.opt) ]")

    if opt.get('model'):

        model_class = load_agent_module(opt['model'])
        # if we want to load weights from --init-model, compare opts with
        # loaded ones
        compare_init_model_opts(opt, opt)
        model = model_class(opt)
        if requireModelExists and hasattr(model, 'load') and not opt.get('model_file'):
            # double check that we didn't forget to set model_file on loadable model
            print('WARNING: model_file unset but model has a `load` function.')
        return model
    else:
        raise RuntimeError('Need to set `model` argument to use create_agent.')

def interactive(args, opt, print_parser=None):
    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: interactive should be passed opt not Parser ]')
        opt = opt.parse_args()

    print("interact check 1 !!!!!!!")
    print("args check!!!!!!!!!!!!!", args)
    # Create model and assign it to the spepython projects/controllable_dialogue/interactiveDemo.py -mf models:controllable_dialogue/control_avgnidf10b10e -wd extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20 --set-controls avg_nidf:7cified task
    agent = create_agent(opt, requireModelExists=True)

    print("interact check 2 !!!!!!!")
    human_agent = LocalHumanAgent(opt)
    print("interact check 3 !!!!!!!")
    world = create_task(opt, [human_agent, agent])
    print("interact check 4 !!!!!!!")
    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Show some example dialogs:
    if not opt.get('chat_script'):
        '''for chateval script evaluation'''
        while True:
            # for conversation in terminal
            # world.parley()
            # for demo
            # print("!!", args)
            # args = 'hello'
            bot_message = world.parley_demo(args)
            # print("bot", bot_message)
            # world.parley_demo(str(args))
            if opt.get('display_examples'):
                print("---")
                print(world.display())
            if world.epoch_done():
                print("EPOCH DONE")
                break
            return bot_message
    else:
        model = get_model_name(opt)
        while True:
            world.parley_script(opt.get('script_input_path'), opt.get('script_output_path'), str(model))
            if opt.get('display_examples'):
                print("---")
                print(world.display())
            if world.epoch_done():
                print("EPOCH DONE")
                break




if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    # interactive(parser.parse_args(print_args=False), print_parser=parser)


