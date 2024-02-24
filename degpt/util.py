import re
import os
import logging
import math
from typing import List, Tuple, Dict, Optional
import tiktoken
from sentence_transformers import SentenceTransformer, util
from cinspector.interfaces import CCode
from cinspector.nodes import CallExpressionNode, BasicNode


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(FILE_DIR, 'log.log')

class Diff:
    """
    Used to locate how chatgpt changes the code

    Attributes:
        src: original code
        tgt: changed code (by chatgpt)
    """

    def __init__(self, src: str, tgt: str):
        import difflib
        self.src = src
        self.tgt = tgt
        self.d = difflib.Differ()

    def diff(self) -> List[str]:
        return self.d.compare(self.src.splitlines(), self.tgt.splitlines())


def halstead_metric(code: str) -> Optional[Dict[str, float]]:
    """
    calculate Halstead Complexity Metrics

    n1 = the number of distinct operators
    n2 = the number of distinct operands
    N1 = the total number of operators
    N2 = the total number of operands

    From these numbers, eight measures can be calculated:

    Program vocabulary: n = n1 + n2
    Program length: N = N1 + N2
    Calculated program length: N'=n1log2(n1)+n2log2(n2)
    Volume: V= Nlog2(n)
    Difficulty: D= (n1/2) * (N2/n2)
    Effort: E= DV
    Time required to program: T= E/18 seconds
    Number of delivered bugs: B=V/3000
    """

    operators = ['+', '-', '*', '/', '^', '&', '%', '=', '==', '!=', '>', '>=', '<', '<=', '&&', '|', '||', '!', '++', '--', '+=', '-=', '*=', '/=', '%=']
    operators += ['<<', '>>', '~', '(', ')', '{', '}', '[', ']', ';', ',', '"', "'", '^', ':', '|=', '?', '^=']
    operators += [
        'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
        'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
        'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
        'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'
    ]
    operators += ['primitive_type', 'type_identifier']

    operands = ['identifier', 'number_literal', 'string_literal', 'statement_identifier', 'string_content', 'character', 'true', 'false', 'null']

    ignore = ['comment', 'escape_sequence']


    def is_operator(n: BasicNode) -> bool:
        if n.node_type in operators:
            return True
        return False

    def is_operand(n: BasicNode) -> bool:
        if n.node_type in operands:
            return True
        return False

    try:
        cc = CCode(code)
    except Exception as e:
        print(e)
        return None

    tokens = cc.node.tokenize()
    operator_lst = []
    operand_lst= []
    for token in tokens:
        if is_operator(token):
            operator_lst.append(token)
        elif is_operand(token):
            operand_lst.append(token)
        elif token.node_type in ignore:
            continue
        else:
            print(f'unknown node type {token.node_type} {token} \n {code} \n ----- \n')
            assert (False and f"Unknown node type")

    dic = dict()
    dic['n1'] = len(set([_.src for _ in operator_lst]))
    dic['n2'] = len(set([_.src for _ in operand_lst]))
    dic['N1'] = len(operator_lst)
    dic['N2'] = len(operand_lst)
    dic['program_vocabulary'] = dic['n1'] + dic['n2']
    dic['program_length'] = dic['N1'] + dic['N2']
    dic['calculated_program_length'] = dic['n1'] * math.log2(dic['n1']) + dic['n2'] * math.log2(dic['n2'])
    dic['volume'] = dic['program_length'] * math.log2(dic['program_vocabulary'])
    dic['difficulty'] = (dic['n1'] / 2) * (dic['N2'] / dic['n2'])
    dic['effort'] = dic['volume'] * dic['difficulty']
    dic['time_required_to_program'] = dic['effort'] / 18
    dic['number_of_delivered_bugs'] = dic['volume'] / 3000
    return dic


def is_code_in_response(code: str, response: str) -> bool:
    """check whether the code is in the response

    Mainly used for ADD_COMMENT optimization
    """

    try:
        cc = CCode(response)
        if cc.get_by_type_name('function_definition'):
            return True
    except Exception as e:
        print(e)

    return False


def response_filter(response: str) -> str:
    """
    filter out ```c and ```. which usually appears in
    the response of structure simplification
    """

    def _filter(_str: str, _target: str) -> str:
        return _str.replace(_target, '')

    response = _filter(response, '```c')
    response = _filter(response, '```C')
    response = _filter(response, '```')
    return response


class Log:
    """
    wrapper of logging module
    """

    def __init__(self, log_file=LOG_FILE, file_level=logging.DEBUG, console_level=logging.INFO) -> None:
        self.log_file = log_file
        self.file_level = file_level
        self.console_level = console_level

    def get(self, name: str):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.log_file, 'a')
        file_handler.setLevel(self.file_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)

        date_fmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt=date_fmt)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger


logger = Log().get(__file__)


# https://platform.openai.com/docs/guides/chat/introduction
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def all_varnames(func_src: str) -> Optional[List[str]]:
    """
    return variable names in function parameter and body
    """
    cc = CCode(func_src)
    func = cc.get_by_type_name('function_definition')
    if len(func) != 1:
        logger.warn(f"Fail to collect all variable names in the following function: \n{func_src}")
        return None
    func = func[0]
    ids = [_ for _ in func.decendants_by_type_name('identifier')]
    block_parent = ['call_expression', 'function_declarator']
    ids = [_.src for _ in ids if _.parent.type not in block_parent]
    return list(set(ids))


def all_invocations(func_src: str) -> Optional[List[str]]:
    """
    read in the source code and return a list
    containing all the invoked function names
    """
    cc = CCode(func_src)
    calls: List[CallExpressionNode] = cc.get_by_type_name('call_expression')
    return [_.function.src for _ in calls if _.is_indirect()]


class Evaluation:
    def __init__(self) -> None:
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def pair_similarity(self, sent1: str, sent2: str) -> float:
        """
        calculate the similarity between sent1 and sent2
        """
        # first we use ProcessName to process the naming style
        pn = ProcessName()
        sent1 = ' '.join(pn.process_name_style(sent1))
        sent2 = ' '.join(pn.process_name_style(sent2))
        embedding1 = self.model.encode(sent1, convert_to_tensor=True)
        embedding2 = self.model.encode(sent2, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(embedding1, embedding2).item())

    def best_similarity(self, name: str, lst: List[str]) -> tuple:
        """
        Given name and lst, calculate which entry in lst has the highest
        similarity with name.

        Return:
            A tuple containing the entry and similarity
        """
        best = -1
        content = None
        for _ in lst:
            similarity = self.pair_similarity(name, _)
            if similarity > best:
                best = similarity
                content = _
        return (content, best)

    """
    Currently, we define several simple and direct evaluation indicators
    """

    def eva_best_average(self, cur_name: List[str], benchmark: List[str]) -> Tuple[float, Dict]:
        """
        For each entry in cur_name, we get its highest similarity compared
        with the entries in benchmark. Then we calculate the average value
        of these highest similarity values and return it.
        """

        similarity_sum = 0
        best_dic = dict()
        for _c in cur_name:
            content, best = self.best_similarity(_c, benchmark)
            similarity_sum += best
            best_dic[_c] = content

        return (similarity_sum / len(cur_name), best_dic)


class ProcessName:
    """
    Used to process the variale name
    """

    def __init__(self) -> None:
        pass

    def complement(self, name: list) -> str:
        """
        complement() is used to complement the abbr

        we will enroll a better one later
        """
        complement_dic = {
            'cur': 'current',
            'prev': 'previous',
        }
        name = [complement_dic[_] if _ in complement_dic.keys() else _ for _ in name]
        return name

    def filter(self, name: list) -> list:
        """
        filter the element like number in the variable name
        """
        re_block = [r'\d+']

        rtn = []
        for _ in name:
            if not any([re.match(_r, _) for _r in re_block]):
                rtn.append(_)
        return rtn

    def split_underscore(self, name: list) -> list:
        rtn_lst = []
        for _ in name:
            rtn_lst += _.split('_')
        return rtn_lst

    def split_uppercase(self, name: list) -> list:
        rtn_lst = []
        for _n in name:
            # if _n = AbcDe, _n_lst = [Abc, De]
            _n_lst = []
            start_index = 0
            for _i in range(1, len(_n)):
                if _n[_i].isupper():
                    _n_lst.append(_n[start_index:_i])
                    start_index = _i
            _n_lst.append(_n[start_index:])

            rtn_lst += _n_lst
        rtn_lst = [_.lower() for _ in rtn_lst]
        return rtn_lst

    def process_name_style(self, name: str) -> list:
        name = [name]
        split_name = self.split_uppercase(self.split_underscore(name))
        return self.filter(self.complement(split_name))


if __name__ == '__main__':
    """
    msg = [
        {
            "role": "system",
            "content": "You provide programming advice.",
        },
        {
            "role": "user",
            "content": "Hi",
        },
    ]
    print(num_tokens_from_messages(msg))
    """

    code = """
    undefined4 main(void) { int a = 0; a >> 2; if(!a) return 0; a+= 1; return a;}
    """

    cc = CCode(code)
    tokens = cc.node.tokenize()
    for _ in tokens:
        print(_.node_type)
    print(tokens)

    print(halstead_metric(code))
