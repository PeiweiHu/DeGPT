"""
We define three roles in this file.

advisor: output the specific advice, how to change
operator: partially adopt the advice from advisor, ensure the original semantic
referee: comment on the changed code, provide next step for refinement
"""


import os
import re
import sys
import traceback
import json
import argparse
import signal
from enum import Enum, unique
from typing import Optional, Dict, List, Tuple
from cinspector.interfaces import CCode
from cinspector.nodes import Util

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(DIR, '.'))
from util import Log, is_code_in_response, response_filter
from mssc import SemanticComparison
from chat import QueryChatGPT, llm_configured, load_config


logger = Log().get(__file__)


PROMPT_PATH = os.path.join(DIR, 'prompt.json')  # file path storing the prompts


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


def is_valid_json(data: str) -> bool:
    """ check whether the data is in json format """

    try:
        json.loads(data)
    except ValueError:
        return False
    return True


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

    perhaps add more optimization types...
    """

    ADD_COMMENT = 'ADD_COMMENT'
    RENAME_VAR = 'RENAME_VAR'
    SIMPLIFY = 'SIMPLIFY'
    ALL = 'ALL'


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

    def __init__(self):

        self.dtype_mapping = {
            DType.ADD_COMMENT: self._add_comment,
            DType.RENAME_VAR: self._rename_var,
            DType.SIMPLIFY: self._simplify,
        }

    def get_advice(self, code: str, dtype: DType) -> Tuple[str, Optional[str]]:
        """ Let chatgpt give detailed modification method for code based on dtype

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

    @staticmethod
    def _replace_variable_name(old_new_dic, code) -> str:
        cc = CCode(code)
        ids = cc.get_by_type_name('identifier')
        old_names = list(old_new_dic.keys())

        for id in ids:
            s_pos = Util.point2index(code, id.start_point[0], id.start_point[1])
            assert (s_pos)
            e_pos = Util.point2index(code, id.end_point[0], id.end_point[1])
            assert (e_pos)
            if str(id) in old_names:
                code = code[:s_pos] + old_new_dic[str(id)] + code[e_pos:]
                return code
        return code

    @staticmethod
    def replace_variable_name(old_new_dic, code) -> str:
        last_code = None
        while last_code != code:
            last_code = code
            code = Advisor._replace_variable_name(old_new_dic, code)

        return code

    def _rename_var(self, code: str, response: Optional[str] = None) -> Tuple[str, str]:
        """
        Rename the variables in the code

        the param <response> is used for replay
        """

        prompt = get_prompt('rename_var', 'advisor')
        assert (prompt)
        if not response:
            q = QueryChatGPT()
            q.insert_system_prompt('You provide the programming suggestions.')
            response = q.query(prompt['content'].format(code=code))

        assert(isinstance(response, str))

        """
        TODO (magic): we do replacement since the json like {'a': 'b'} is treated as invalid
        """
        if "':" in response:
            response = response.replace("'", '"')

        if not is_valid_json(response):
            logger.warning(f"Fail to rename variables since the response is not valid JSON: {response}")
            return (code, response)

        old_new_dic = json.loads(response)

        try:
            code = self.replace_variable_name(old_new_dic, code)
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
        q = QueryChatGPT()
        q.insert_system_prompt('You provide the programming suggestions.')
        response = q.query(prompt['content'].format(code=code))
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
        q = QueryChatGPT()
        q.insert_system_prompt('You provide the programming suggestions.')
        response = q.query(prompt['content'].format(code=code))
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
    def __init__(self):
        pass

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
            return advised_code
        elif dtype == DType.SIMPLIFY:
            return advised_code

            """ TODO: rewrite semantic comparison for better accuracy
            semantic_cmp = SemanticComparison(original_code, advised_code)
            if run_timer(semantic_cmp.is_semantic_equal, time=10):
            # if semantic_cmp.is_semantic_equal():
                return advised_code
            else:
                return original_code
            """
        else:
            logger.warning(f"The operator on {dtype} is not implemented, skip this change")

        return original_code


class Referee(Role):
    """
    Comment on the changed code, provide next step for refinement.

    Attributes:
        cm: the chat manager

    Methods:
        get_direction: get the direction for the code change
    """
    def __init__(self):
        pass

    def get_direction(self, code: str) -> Tuple[str, List[DType]]:
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
        q = QueryChatGPT()
        q.insert_system_prompt('You provide the programming suggestions')
        response = q.query(prompt['content'].format(code=code))
        assert (isinstance(response, str))
        logger.info('[Referee] response: {}'.format(response))
        directions = self._parse_need(response)
        return (response, directions)

    def _parse_need(self, response: str) -> List[DType]:
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


def single_opt(decompile_code: str, opt_type: DType) -> dict:
    """ execute single optimization assigned by <opt_type> on <decompile_code> """

    dic = {'decompiler_output': decompile_code}
    advisor = Advisor()
    advisor_code, response = advisor.get_advice(decompile_code, opt_type)
    operator = Operator()
    operator_code = operator.operate(decompile_code, advisor_code, opt_type)
    dic['output'] = operator_code
    return dic


class RoleModel:
    """
    manage the workflow of the three-role model
    """

    def __init__(self, *, decompile_code: Optional[str] = None, src_code: Optional[str] = None):
        """
        Args:
            decompile_code: decompiler output of the function
            src_code: the source code of the function (used for evaluation, optional)
        """

        self.code = decompile_code
        self.src_code = src_code
        self.advisor = Advisor()
        self.operator = Operator()
        self.referee = Referee()

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

    @staticmethod
    def sub_wf(wf1: str, wf2: str) -> int:
        """ whether wf1 is the later step (or same) of wf2, wf - workflow """
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
        res['output'] = get_optimized_from_dic(res)
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


def opt_str2dtype(opt_type: str) -> DType:

    mapping = {
        'rename': DType.RENAME_VAR,
        'simplify': DType.SIMPLIFY,
        'comment': DType.ADD_COMMENT,
        'all': DType.ALL,
    }

    return mapping[opt_type]


def single_run(decompile_code: str, output: str, opt_type: str) -> None:

    assert (opt_type in ['rename', 'simplify', 'comment', 'all'])

    try:
        if opt_type != 'all':
            # conduct single optimization
            dic = single_opt(decompile_code, opt_str2dtype(opt_type))
        else:
            model = RoleModel(decompile_code=decompile_code)
            dic = model.work()
    except Exception as e:
        logger.warning(f"Fail to run due to {e}")
        print(traceback.format_exc())
        return

    with open(output, 'w') as w:
        json.dump(dic, w, indent=4)

    print('='*10 + 'after optimization' + '='*10)
    print(dic['output'])


def single_run_file(decompile_file: str, output: str, opt_type: str) -> None:

    def read_code(f: str) -> str:
        with open(f, 'r') as r:
            return r.read()

    single_run(read_code(decompile_file), output, opt_type)


def parse_arguments():
    parser = argparse.ArgumentParser(description='DeGPT: Optimizing Decompiler Output with LLM')
    parser.add_argument('-t', choices = ['rename', 'simplify', 'comment', 'all'], default='all', help='Assign the optimization type')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--string', nargs=2, metavar=('decompiler_str', 'output_json'), help='Optimize the decompiler_str')
    group.add_argument('-f', '--file', nargs=2, metavar=('decompiler_file', 'output_json'), help='Optimize the content of the file decompiler_file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    if not llm_configured():
        print('please complete llm access setup first...')
        exit()

    args = parse_arguments()
    if args.string:
        single_run(args.string[0], args.string[1], args.t)
    elif args.file:
        single_run_file(args.file[0], args.file[1], args.t)
