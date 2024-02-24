"""
We define three roles in this file.

advisor: output the specific advice, how to change
operator: partially adopt the advice from advisor, ensure the original semantic
referee: comment on the changed code, provide next step for refinement


q1: Should they share the same session? I will setup a parameter in the constructor,
so decide this later.

q2: We should do a statistic analysis to decide what kind of changes may be suggested
by referee, and what kind of specific advice may be suggested by advisor focusing on
the suggestion of referee.

q3: update the bare_query-related stuffs. We wanna store all prompts in it.

"""


import os
import re
import sys
import ctypes
import random
import json
import argparse
import signal
import threading
from concurrent import futures
from enum import Enum, IntEnum, unique
from typing import Optional, Dict, List, Tuple
from cinspector.interfaces import CCode
from chatmanager import ChatManager, ChatMessage, ChatResponse, ChatSetup

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(DIR, '.'))
from util import Evaluation, Log, is_code_in_response, response_filter
from mssc import SemanticComparison


logger = Log().get(__file__)


# ------------------- config here ---------------------

PROMPT_PATH = os.path.join(DIR, 'prompt.json')
CHATGPT_API_KEY = None
ChatSetup.temperature = 0.2
assert (CHATGPT_API_KEY and "Pleae set api key")
# -----------------------------------------------------


def run_timer(func, *, args = [], time = 1, info = 'run_timer failed'):
    """
    timer - limit the running time of the func
    """

    def timeout_callback(signum, frame):
        raise Exception(f'timeout')

    signal.signal(signal.SIGALRM, timeout_callback)
    signal.alarm(time)
    try:
        rtn = func(*args)
        signal.alarm(0)
        return rtn
    except Exception as e:
        print(e)
        print(info)
        return None


def get_prompt(name: str, _type: str, prompt_path: str = PROMPT_PATH) -> Optional[Dict[str, str]]:
    """
    Access the prompt

    Args:
        name: the name of the prompt
        _type: the type of the prompt
        prompt_path: the path of the prompt file

    Returns:
        a dict containing two keys: 'role' and 'content'
    """
    import json

    prompts = None
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
    assert(prompts)

    for _p in prompts:
        if _p['name'] == name and _p['type'] == _type:
            return _p['prompt']
    return None


@unique
class DType(str, Enum):
    """
    The types of the directions from referee.
    """

    ADD_COMMENT = 'ADD_COMMENT'
    RENAME_VAR = 'RENAME_VAR'
    SEGMENT = 'SEGMENT'
    FORMAT = 'FORMAT'
    RENAME_FUNC_PARA = 'RENAME_FUNC_PARA'
    SIMPLIFY = 'SIMPLIFY'


class Role:
    """
    The base class of three roles: referee, advisor, operator
    """

    def __init__(self):
        pass


class Advisor(Role):
    """
    Provide specific advice for the code change.

    Attributes:
        cm: the chat manager
        dtype_mapping: the mapping from direction type to the corresponding process function

    Methods:
        get_advice(code: str, dtype: DType): Let chatgpt give detailed modification method for code based on dtype
    """

    def __init__(self, cm: ChatManager = ChatManager()):
        self.cm: ChatManager = cm
        # self.cm.add_chat('system', 'You provide the programming suggestions.')

        self.dtype_mapping = {
            DType.ADD_COMMENT: self._add_comment,
            DType.RENAME_VAR: self._rename_var,
            # DType.RENAME_FUNC_PARA: self._rename_func,
            DType.SIMPLIFY: self._simplify,
        }

    def get_advice(self, code: str, dtype: DType) -> Tuple[str, Optional[str]]:
        """Let chatgpt give detailed modification method for code based on dtype

        This function first finds the corresponding processing method from
        self.dtype_mapping based on dtype, then calls the method to get
        the new code and returns it.

        Args:
            code: the code waiting for advice
            dtype: the type of the advice (from referee)
        Returns:
            a tuple containing the new advised code and the response from ChatGPT
        """

        if dtype not in self.dtype_mapping.keys():
            logger.warning(f"Fail to get the processing method for the dtype {dtype}, skip")
            return (code, None)  # return original code, no change

        method = self.dtype_mapping[dtype]
        code, response = method(code)
        logger.info(f"[Advisor] {'='*10} response for {dtype} {'='*10} \n {response} \n {'='*20}")
        return (code, response)

    """ ============= Define the processing methods in the following ============== """

    def _rename_var(self, code: str, response: Optional[str] = None) -> Tuple[str, str]:
        """
        Rename the variables in the code

        the param <response> is used for replay
        """

        prompt = get_prompt('rename_var', 'advisor')
        assert (prompt)
        if not response:
            cmsg = ChatMessage()
            cmsg.push_system('You provide the programming suggestions.')
            cmsg.push_user(prompt['content'].format(code=code))
            response = self.cm.send(cmsg).get_msg()

        assert(isinstance(response, str))

        """
        we do replacement since the json like {'a': 'b'} is treated as invalid
        """
        if "':" in response:
            response = response.replace("'", '"')

        import json
        # the response is excepted to be in JSON format
        def is_valid_json(data: str) -> bool:
            try:
                json.loads(data)
            except ValueError:
                return False
            return True

        if not is_valid_json(response):
            logger.warning(f"Fail to rename variables since the response is not valid JSON: {response}")
            return (code, response)

        # do replacement, currently we perform very simple replacement
        old_new_dic = json.loads(response)
        try:
            """ get all invocation name """
            cc = CCode(code)
            call_exps = cc.get_by_type_name('call_expression')
            callees = []
            for exp in call_exps:
                if exp.function:
                    callees.append(exp.function.src)

            for k, v in old_new_dic.items():
                # avoid rename the callee
                if k in callees:
                    continue
                code = code.replace(k, v)
        except Exception as e:
            logger.warning(e)
            return (code, response)
        return (code, response)

    def _add_comment(self, code: str) -> Tuple[str, str]:
        """
        Add comment to explain the purpose
        """
        prompt = get_prompt('add_comment', 'advisor')
        assert(prompt)
        cmsg = ChatMessage()
        cmsg.push_system('You provide the programming suggestions.')
        cmsg.push_user(prompt['content'].format(code=code))
        response = self.cm.send(cmsg)
        assert (isinstance(response, ChatResponse))
        response = response.get_msg()
        assert (isinstance(response, str))
        response = response_filter(response)
        if not is_code_in_response(code, response):
            response = f"\\*{response}*\\ \n \n{code}"
        return (response, response)

    def _simplify(self, code: str) -> Tuple[str, str]:
        """
        Simplify the code
        """
        prompt = get_prompt('remove_unnecessary', 'advisor')
        assert(prompt)
        cmsg = ChatMessage()
        cmsg.push_system('You provide the programming suggestions.')
        cmsg.push_user(prompt['content'].format(code=code))
        response = self.cm.send(cmsg)
        assert (isinstance(response, ChatResponse))
        response = response.get_msg()
        assert (isinstance(response, str))
        response = response_filter(response)
        # here we extract the function to avoid to additional info from ChatGPT
        try:
            response = CCode(response).get_by_type_name('function_definition')[0].src
        except Exception as e:
            print(e)
        return (response, response)


class Operator(Role):
    """
    Adopt the advice from advisor, ensure the original semantic.
    """
    def __init__(self, cm: ChatManager = ChatManager()):
        self.cm: ChatManager = cm

    def operate(self, original_code: str, advised_code: str, dtype: DType) -> str:
        """
        Adopt the advice from advisor, ensure the original semantic.

        Args:
            original_code: the original code
            adivsed_code: new code advised by advisor
            dtype: the change type

        Returns:
            the new code, some unreasonable changes of advised_code may be discarded
        """

        if dtype == DType.ADD_COMMENT:
            return advised_code
        elif dtype == DType.RENAME_VAR:
            # currently no check
            return advised_code
        elif dtype == DType.RENAME_FUNC_PARA:
            # currently no check
            return advised_code
        elif dtype == DType.SIMPLIFY:
            semantic_cmp = SemanticComparison(original_code, advised_code)
            if run_timer(semantic_cmp.is_semantic_equal, time=10):
            # if semantic_cmp.is_semantic_equal():
                return advised_code
            else:
                return original_code
        else:
            logger.warning(f"The operate on {dtype} is not implemented, skip this change")

        return original_code


class Referee(Role):
    """
    Comment on the changed code, provide next step for refinement.

    Attributes:
        cm: the chat manager

    Methods:
        get_direction: get the direction for the code change
    """
    def __init__(self, cm: ChatManager = ChatManager()):
        self.cm: ChatManager = cm

    def get_direction(self, code: str) -> Tuple[str, List[Optional[DType]]]:
        """
        Get the direction for the code change.

        Args:
            code: the code waiting for comment
        Returns:
            A tuple containing the response from ChatGPT and the list of directions (DType)
        """
        prompt = get_prompt('need', 'referee')
        assert(prompt)

        # complement the prompt with the code
        cmsg = ChatMessage()
        cmsg.push_system('You provide the programming suggestions.')
        cmsg.push_user(prompt['content'].format(code=code))
        response = self.cm.send(cmsg).get_msg()
        # print(response)
        assert (isinstance(response, str))
        logger.info('[Referee] response: {}'.format(response))
        # directions = self._parse_direction(code, response)
        directions = self._parse_need(code, response)
        return (response, directions)

    def _construct_dtype_template(self) -> Dict[DType, List[str]]:
        """
        construct the template for the dtype for the similarity calculation
        """
        dtype_template = dict()
        # ADD_COMMENT
        dtype_template[DType.ADD_COMMENT] = [
                'Add comment to explain the purpose',
            ]
        # RENAME_VAR
        dtype_template[DType.RENAME_VAR] = [
                'Rename variables to more descriptive names',
                'Use meaningful variable names that reflect the purpose of the variable and make the code easier to understand.',
            ]
        # SEGMENT
        dtype_template[DType.SEGMENT] = [
                # 'Break up the code into smaller functions',
                'this is currently forbidden',
            ]
        # FORMAT
        dtype_template[DType.FORMAT] = [
                'Use consistent and clear formatting throughout the code',
            ]
        # RENAME_FUNCN_PARA
        dtype_template[DType.RENAME_FUNC_PARA] = [
                # 'Use meaningful function and parameter names',
                'this is currently forbidden',
            ]
        # SIMPLIFY
        dtype_template[DType.SIMPLIFY] = [
                'Simplify the function by removing unnecessary code and comments',
                'Break down the function into smaller, more manageable functions that carry out specific tasks.'
                'Break up the code into smaller functions',
            ]

        return dtype_template

    def _conclude_dtype(self, suggestion: str) -> Optional[DType]:
        """
        read in the suggestion from referee and output the dtype it belongs to
        """

        threshold = 0.4
        max_sim = 0
        dtype = None
        eva = Evaluation()

        for _k, _v in self._construct_dtype_template().items():
            for _sent in _v:
                sim = eva.pair_similarity(_sent, suggestion)
                # logger.debug('[Referee] calculate sim: {}\n{}\n{}\n'.format(sim, _sent, suggestion))
                if sim > max_sim:
                    max_sim = sim
                    dtype = _k

        if max_sim < threshold:
            return None
        return dtype

    def _parse_need(self, code: str, response: str) -> List[DType]:
        rtn = []
        pattern = r'\b(?:Yes|yes|No|no)\b'
        matches = re.findall(pattern, response)
        assert (len(matches) == 3)
        if matches[0] in ['Yes', 'yes']:
            rtn.append(DType.SIMPLIFY)
        if matches[1] in ['Yes', 'yes']:
            rtn.append(DType.ADD_COMMENT)
        if matches[2] in ['Yes', 'yes']:
            rtn.append(DType.RENAME_VAR)
        return rtn

    def _parse_direction(self, code: str, response: str) -> List[Optional[DType]]:
        """
        iterate each suggestion and classify it into a dtype by
        similarity calculation
        """

        dtype_lst = list()
        lines = response.split('\n')
        for _l in lines:
            _l = _l.strip()
            dtype = self._conclude_dtype(_l)
            dtype_lst.append(dtype)
        return dtype_lst


class RoleModel:
    """
    manage the workflow of the three-role model
    """

    def __init__(self, *, decompile_code: Optional[str] = None, src_code: Optional[str] = None, cm: ChatManager = ChatManager()):
        """
        Args:
            code: decompiler output of the function
            src_code: the source code of the function
            cm: ChatManager
        """

        self.code = decompile_code
        self.src_code = src_code
        self.cm = cm
        if not self.cm.keys.key_name_exist('key1'):
            self.cm.add_key('key1', CHATGPT_API_KEY)
        self.cm.set_session('default')
        self.advisor = Advisor(self.cm)
        self.operator = Operator(self.cm)
        self.referee = Referee(self.cm)

    def save_session(self, path: str):
        exported_json_str = self.cm.export_session()
        assert (isinstance(exported_json_str, str))
        with open(path, 'w') as w:
            w.write(exported_json_str)

    def sort_directions(self, direction_lst: List[DType]) -> List[str]:
        """
        remove None and uninterested dtype, sort the left
        directions by the priority
        """

        sort_index = {
            DType.SIMPLIFY: 0,  # highest priority
            DType.ADD_COMMENT: 0.5,
            DType.RENAME_VAR: 1,
        }

        sorted_directions = list()
        # sort the directions based on sort_index and put the result in sorted_directions
        directions = set(direction_lst)
        for _d in directions:
            # filter out None and uninterested DType
            if _d is None or sort_index[_d] == -1:
                continue

            if not sorted_directions:
                sorted_directions.append(_d)
                continue

            for _i, _sd in enumerate(sorted_directions):
                if sort_index[_d] < sort_index[_sd]:
                    sorted_directions.insert(_i, _d)
                    break
                if _i == len(sorted_directions) - 1:
                    sorted_directions.append(_d)
                    break

        return sorted_directions

    def __del__(self):
        self.save_session('session.json')

    @staticmethod
    def sub_wf(wf1: str, wf2: str) -> int:
        """ whether wf1 is the later step (or same) of wf2 """
        dic = {
            'INIT': 0,
            'REFEREE': 1,
            'OPT:SIMPLIFY': 2,
            'DONE': 3,
        }

        return dic[wf1] - dic[wf2]

    @staticmethod
    def restore_to(workflow: str, existing_json: str, output: Optional[str] = None):
        """
        readin the existing_json and restore it to <workflow>, then write
        the new dic to output (original existing_json as default)

        for example, resotre the dic whose workflow status is
        DONE to OPT:SIMPLIFY
        """

        assert (workflow in ['INIT', 'REFEREE', 'OPT:SIMPLIFY', 'DONE'])

        r = open(existing_json, 'r')
        res = json.load(r)
        r.close()
        cur_workflow = res['workflow']

        # if the target workflow is the later step of the cur_workflow, skip directly
        if RoleModel.sub_wf(workflow, cur_workflow) >= 0:
            print(f"Skip {existing_json} (workflow: {cur_workflow})")
            return

        # require restore to the status before DONE (i.e., OPT:SIMPLIFY, REFEREE, INIT)
        if RoleModel.sub_wf(workflow, 'DONE') < 0:
            # remove the optimizations after SIMPLIFY
            if 'SIMPLIFY' in res['optimization'].keys():
                res['optimization'] = {'SIMPLIFY': res['optimization']['SIMPLIFY']}
            else:
                res['optimization'] = dict()

        if RoleModel.sub_wf(workflow, 'OPT:SIMPLIFY') < 0:  # REFEREE or INIT
            # remove the optimization SIMPLIFY
            res['optimization'].clear()

        if RoleModel.sub_wf(workflow, 'REFEREE') < 0:  # INIT
            # remove optimization dic and direction-related things
            for _ in ['optimization', 'sorted_directions', 'original_directions', 'original_directions_src']:
                res.pop(_)

        res['workflow'] = workflow
        out = output if output else existing_json
        with open(out, 'w') as w:
            print(f"Restore {existing_json} (workflow: {cur_workflow}) to {workflow} and dump to {out}")
            json.dump(res, w, indent=4)

    def work(self, end_at: str = 'DONE', existing_json: Optional[str] = None):
        """
        self.work() will return a dict describing the
        status of the refinement.
        {
            'workflow': str, # REFEREE,
            'source_code': str,
            'decompiler_output': str,
            'original_directions_src': str,
            'original_directions': DType,
            'sorted_directions': DType,
            'optimization': {
                'optimization type name': {
                    'input': 'input code',
                    'output': 'output code',
                    'status': 'SUCC or FAIL',
                    'append': 'any other comment',
                    'advisor': 'reponse from advisor',
                    'operator': 'reponse from operator',
                },
                ...... ,
            }
        }


        To facilitate the analysis, we design the stop-recover mechanism
        for the workflow. The field 'workflow' in the output json file lables
        the current workflow status.

            INIT - Nothing is done yet.
            REFEREE - The role referee ends.
            OPT:SIMPLIFY - The SIMPLIFY optimization is done.
            DONE - All done.
        """

        if existing_json:
            res = None
            with open(existing_json, 'r') as r:
                res = json.load(r)
            # if the loaded json records the work that already runs end_at
            if self.sub_wf(res['workflow'], end_at) >= 0:
                return res
        else:
            res = dict()
            res['source_code'] = self.src_code
            res['decompiler_output'] = self.code
            res['workflow'] = 'INIT'

        # check user assigned end_at
        if self.sub_wf(end_at, 'INIT') <= 0:
            return res
        else:
            print('pass INIT checking')

        # the role referee starts working
        if self.sub_wf('REFEREE', res['workflow']) > 0:
            response, directions = self.referee.get_direction(res['decompiler_output'])
            res['original_directions_src'] = response
            res['original_directions'] = directions
            logger.info(f'[RoleModel] directions: {directions}')

            directions = self.sort_directions(directions)
            res['sorted_directions'] = directions
            logger.info(f'[RoleModel] sorted directions: {directions}')

            res['optimization'] = dict()
            res['workflow'] = 'REFEREE'

        # check user assigned end_at
        if self.sub_wf(end_at, res['workflow']) == 0:
            return res

        if 'SIMPLIFY' not in res['sorted_directions'] and self.sub_wf('OPT:SIMPLIFY', res['workflow']) > 0:
            res['workflow'] = 'OPT:SIMPLIFY'

        # start checking OPT:SIMPLIFY
        if self.sub_wf(end_at, res['workflow']) == 0:
            return res

        for _direction in res['sorted_directions']:

            # Skip if OPT:SIMPLIFY is already done. Mainly used for existing_json.
            if _direction == 'SIMPLIFY' and self.sub_wf('OPT:SIMPLIFY', res['workflow']) <= 0:
                continue

            optimization = dict()
            res['optimization'][_direction] = optimization

            # input is the output of the last optimization
            dindex = res['sorted_directions'].index(_direction)
            if dindex == 0:
                optimization['input'] = res['decompiler_output']
            else:
                # print(res['sorted_directions'])
                # print(res['sorted_directions'][dindex - 1])
                optimization['input'] = res['optimization'][res['sorted_directions'][dindex - 1]]['output']

            """
            <adviced_code> is the suggested code from advisor, <response> is the direct resposne that advisor
            gets from ChatGPT. For add_comment and structure simplification, they are the smae.
            """
            adviced_code, response = self.advisor.get_advice(optimization['input'], _direction)
            optimization['advisor'] = adviced_code
            optimization['advisor_response'] = response
            if adviced_code == optimization['input']:
                optimization['status'] = 'FAIL|ADVISOR'
                # check SIMPLIFY
                if _direction == 'SIMPLIFY' and self.sub_wf('OPT:SIMPLIFY', res['workflow']) > 0:
                    res['workflow'] = 'OPT:SIMPLIFY'
                if self.sub_wf(end_at, 'OPT:SIMPLIFY') == 0:
                    return res
                # end check
                continue

            optimization['operator'] = self.operator.operate(optimization['input'], adviced_code, _direction)

            if optimization['operator'] == optimization['input']:
                optimization['status'] = 'FAIL|OPERATOR'
                optimization['output'] = optimization['input']
            else:
                optimization['status'] = 'SUCC'
                optimization['output'] = optimization['operator']


            # check SIMPLIFY
            if _direction == 'SIMPLIFY' and self.sub_wf('OPT:SIMPLIFY', res['workflow']) > 0:
                res['workflow'] = 'OPT:SIMPLIFY'
            if self.sub_wf(end_at, 'OPT:SIMPLIFY') == 0:
                return res
            # end check

        res['workflow'] = 'DONE'
        return res


def replay_advisor_rename(dic_path):
    """
    Given the result dic and optimization type, this function replays the
    workflow to figure out the reason of failure.
    """

    dic = None
    with open(dic_path, 'r') as r:
        dic = json.load(r)
    rename_input = dic['optimization']['RENAME_VAR']['input']
    rename_response = dic['optimization']['RENAME_VAR']['advisor_response']
    advisor = Advisor()
    advisor._rename_var(rename_input, rename_response)


def restore_dir(dir_path: str, workflow: str):
    """
    restore all dic under dir_path to workflow
    """

    for case in os.listdir(dir_path):
        case = os.path.join(dir_path, case)
        RoleModel.restore_to(workflow, case)


def resume_from_dic(dir_path: str, workflow: str):
    """
    read data from the existing dic in dir_path and execute until workflow
    """

    for case in os.listdir(dir_path):
        case = os.path.join(dir_path, case)
        model = RoleModel()
        dic = model.work(workflow, case)
        with open(case, 'w') as w:
            json.dump(dic, w, indent=4)


def get_optimized_from_dic(dic) -> str:
    opts = dic['optimization']
    opt_order = ['SIMPLIFY', 'ADD_COMMENT', 'RENAME_VAR']
    out = dic['decompiler_output']
    for _ in opt_order:
        if opts[_]['status'].startswith('FAIL'):
            return out
        else:
            out = opts[_]['output']
    return out


def single_run(decompile_code: str, output: str) -> None:

    try:
        model = RoleModel(decompile_code=decompile_code)
        dic = model.work()
    except Exception as e:
        logger.warning(f"Fail to run due to {e}")
        return

    with open(output, 'w') as w:
        json.dump(dic, w, indent=4)

    print('='*10 + 'after optimization' + '='*10)
    print(get_optimized_from_dic(dic))


def single_run_file(decompile_file: str, output: str) -> None:

    def read_code(f: str) -> str:
        with open(f, 'r') as r:
            return r.read()

    single_run(read_code(decompile_file), output)


def parse_arguments():
    parser = argparse.ArgumentParser(description='DeGPT: Optimizing Decompiler Output with LLM')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', nargs=2, metavar=('decompiler_str', 'output_json'), help='Optimize the decompiler_str')
    group.add_argument('-f', nargs=2, metavar=('decompiler_file', 'output_json'), help='Optimize the content of the file decompiler_file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    if args.s:
        single_run(args.s[0], args.s[1])
    elif args.f:
        single_run_file(args.f[0], args.f[1])
