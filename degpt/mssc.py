import random
import re
import sys
import copy
import os
import signal
from enum import Enum, unique
from Levenshtein import distance
from typing import List, Optional, Set, Dict, Union
from cinspector.interfaces import CCode
from cinspector.nodes import BasicNode, Node, CallExpressionNode, Util, FunctionDefinitionNode
from cinspector.analysis import CFG

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(DIR, '..'))


def run_timer(func, *, args = [], time = 1, info = 'run_timer failed'):
    def timeout_callback(signum, frame):
        raise Exception(f'timeout')
    signal.signal(signal.SIGALRM, timeout_callback)
    signal.alarm(time)
    try:
        rtn = func(*args)
    except Exception as e:
        print(e)
        print(info)
        return None
    signal.alarm(0)
    return rtn


@unique
class SType(Enum):
    """
    The type of the symbol value
    """

    # number
    NUMBER = 1
    # memory location
    MEMORY = 2
    # unknown
    UNKNOWN = 3


def unpack_call_expression(call_exp: CallExpressionNode) -> Optional[List[str]]:
    """
    parse the call expression and output the callee and arguments

    return None if error happens, otherwise, [callee, arg1, arg2, ...]
    """

    try:
        callee = call_exp.function.src
        arguments = [_.src.strip() for _ in Util.sort_nodes(call_exp.arguments)]
    except Exception as e:
        print(e)
        return None

    return [callee] + arguments


def unpack_and_fill_call_expression(call_exp: CallExpressionNode, sym_tab: 'SymTable', call_tab: 'CallTable') -> Optional[List[Union[str, float]]]:
    """
    While unpack_call_expression simply output the src of callee and arguments,
    this function will try to calculate the arguments.
    """

    call = unpack_call_expression(call_exp)
    if not call:
        return None
    real_call = [call[0]]
    for _ in call[1:]:
        # argument maybe in sym_tab
        if val := sym_tab.get_sym(_):
            real_call.append(val.get_value())
        # argument maybe another call
        elif (arg := CCode(_ + ';').node.children[0]) and arg.children and arg.children[0].node_type == 'call_expression':
            real_arg = unpack_and_fill_call_expression(arg.children[0], sym_tab, call_tab)
            if call_tab.has(real_arg):
                real_call.append(int(call_tab.get_value(real_arg)))
            else:
                real_call.append(_)
        else:
            real_call.append(_)
    return real_call


class Sym:
    """ The symbol value of the symbol table

    helper class of SymTable, mainly used to
    distinguish different types of symbols

    1. number
    2. memory location
    3. unknown

    """

    def __init__(self, sym: Union[str, int], ty: SType) -> None:
        self.sym: Union[str, int] = sym
        self.type: SType = ty

    def get_value(self) -> Union[str, int]:
        if self.type == SType.NUMBER:
            return int(self.sym)
        return self.sym

    def get_type(self) -> SType:
        return self.type

    def __str__(self) -> str:
        return f'{self.sym}'


class SymTable:
    """
    helper class of MicroSnippetSemanticCalulation, used to
    record the symbol and memory location's value.
    """

    def __init__(self, ids: Set[str]):
        self.ids = ids
        self.sym_table: Dict[str, Sym] = dict()
        self._init_value()

    def reset(self):
        self._init_value()

    def dry(self):
        """
        return a dry symtable in dict format
        """
        return {k: v.get_value() for k, v in self.sym_table.items()}

    def _init_value(self):
        """
        Init random values for the symbols
        """
        MIN, MAX = 0, 100000
        value_logger = set()
        for _id in self.ids:
            value = random.randint(MIN, MAX)
            while value in value_logger:
                value = random.randint(MIN, MAX)
            value_logger.add(value)
            assert(isinstance(_id, str))
            self.sym_table[_id] = Sym(value, SType.NUMBER)

    def print_sym(self, banner='Symbol Table'):
        """
        output the current symbol table status in stdout
        """
        banner = '='*15 + banner + '='*15
        print(banner, '\n')
        for _k, _v in self.sym_table.items():
            print(f'{_k}: {_v}')
        print('='*len(banner), '\n')

    def set_sym(self, sym: str, value: Sym):
        # print(f'[SymTable] set {sym} to {value}')
        self.sym_table[sym] = value

    def get_sym(self, sym: str) -> Optional[Sym]:
        if sym not in self.sym_table.keys():
            return None
        # print(f'[SymTable] get {sym} with value {self.sym_table[sym]}')
        return self.sym_table[sym]


class SemanticComparison:
    """
    MicroSnippetSemanticCalulation calculate the
    symbol changes of the single code snippet.
    This class compares the symbol changes of two
    code snippets.
    """

    def __init__(self, code1: str, code2: str):
        self.code1 = code1
        self.code2 = code2
        self.logger = []

    def collect_invocation(self, code_snippet: str) -> List[Optional[List[str]]]:
        """
        the step 1: collect the invocations
        """
        cc = CCode(code_snippet)
        invocations = cc.get_by_type_name('call_expression')
        res = [unpack_call_expression(_) for _ in invocations]
        return res

    def is_call_equal(self) -> bool:
        call1 = self.collect_invocation(self.code1)
        call2 = self.collect_invocation(self.code2)

        def _lst_in(lst1: List[str], lst2: List[str]) -> bool:
            return all([_ in lst2 for _ in lst1])

        return _lst_in(call1, call2) and _lst_in(call2, call1)

    def _dict_equal(self, dict1: Dict, dict2: Dict) -> bool:

        def _dict_in(a, b) -> bool:
            """
            b contains every k-v in a

            Note that since the LLM has the ability of removing redundant
            variables, it's possible that some local variables are non-exist
            after optimization.

            We allow that variables starting with uVar, iVar, local, pvVar,
            puVar only exist in one of the dicts.
            """

            for k, v in a.items():
                if k.endswith('++'):
                    continue
                local_var_lst = ['uVar', 'iVar', 'local', 'pvVar', 'piVar', 'puVar', 'sVar', 'dVar']
                local_var_lst += ['*' + _ for _ in local_var_lst]

                local_var_keyword = ['Var']
                # decompiler-related stuffs
                decompiler_var_lst = ['FS_OFFSET', '__stack_chk_fail']

                # skip local variable, we don't compare local variable since they don't have side effect
                if any([k.startswith(_) for _ in local_var_lst]):
                    continue
                if any([_ in k for _ in local_var_keyword]):
                    continue
                # skip decompiler-related variable
                if any([_ in k for _ in decompiler_var_lst]):
                    continue

                # any ids has distance less than 3?
                similar_ids = [_ for _ in b.keys() if distance(k, _) < 4]
                # print(f'similar_ids for {k} is {similar_ids}')
                if k not in b.keys() and len(similar_ids) == 0:
                    self.logger.append(f'{k} does not exist.')
                    """
                            IMPORTANT
                    """
                    continue
                    # return False

                flag1 = k in b.keys() and v == b[k]
                flag2 = any([v == b[_] for _ in similar_ids])
                if (not flag1) and (not flag2):
                    self.logger.append(f'{k} has different value, {v} {b[k]}')
                    return False

            return True

        return _dict_in(dict1, dict2) and _dict_in(dict2, dict1)

    def is_mssc_equal(self) -> bool:
        mssc1 = MicroSnippetSemanticCalulation(self.code1)
        res_lst1 = mssc1.calculate()
        sym_tab = mssc1.init_sym_table
        call_tab = mssc1.call_table.reference_table
        # print('-----call table------- \n' + str(call_tab) + '\n--------------\n\n')
        mssc2 = MicroSnippetSemanticCalulation(self.code2, sym_tab, call_tab)
        res_lst2 = mssc2.calculate()
        """
        print('='*10)
        for _ in res_lst1:
            print(_)
            print()
        print('='*10)
        for _ in res_lst2:
            print(_)
            print()
        """

        """
        compare res1 and res2, there are multiple possible
        results:

            1. res_lst1 and res_lst2 have differnent length indicates
                two snippets own different execution path

            XXXX

        currently we treat two snippets is semantically equal if their res_lsts
        are one-one mapping
        """

        """
        It's not reasonable to compare the length since LLM has the ability of
        removing redundant variables.

        if len(res_lst1) != len(res_lst2):
            self.logger.append('two snippets have different mssc result length')
            return False
        """

        for res in res_lst1:
            sym_tab, call_tab = res
            flag = False
            for _ in res_lst2:
                cur_sym_tab, cur_call_tab = _
                # first we compare call table
                if not self._dict_equal(call_tab, cur_call_tab):
                    continue
                # then we compare the symbol table
                if self._dict_equal(sym_tab, cur_sym_tab):
                    flag = True
            if not flag:
                self.logger.append(f'two snippets have different mssc value')
                self.logger.append(f'\n\n Fail while searching {res} \n\n')
                return False

        return True

    def is_semantic_equal(self) -> bool:
        """
        if not self.is_call_equal():
            self.logger.append('two snippets are not call equal')
            return False
        """
        try:
            if not self.is_mssc_equal():
                self.logger.append('two snippets are not mssc equal')
                return False
        except Exception as e:
            print(e)
            return False
        return True


class CallTable:

    def __init__(self, reference_table: Optional[Dict[str, float]] = None) -> None:
        """
        repo: {
            call_hash: random float value
        }
        """
        self.repo: Dict[str, float] = dict()
        self.reference_table = reference_table if reference_table else dict()

    def hash_call(self, call: List[str]) -> str:
        s = ''
        for _ in call:
            s += str(_) + ','
        return s

    def add(self, call: List[str]) -> None:
        """
        call -> [callee, arg1, arg2, ...], note that arg1, arg2
        should be the float value (from sym_tab)
        """

        if self.has(call):
            return

        h = self.hash_call(call)
        if h in self.reference_table.keys():
            self.repo[h] = self.reference_table[h]
        else:
            self.repo[h] = float(random.randint(0, 100000))
            self.reference_table[h] = self.repo[h]  # log the newly allocated value

    def has(self, call: List[str]) -> bool:
        return self.hash_call(call) in self.repo.keys()

    def get_value(self, call: List[str]) -> Optional[float]:
        if not self.has(call):
            return None

        return self.repo[self.hash_call(call)]

    def dry(self) -> Dict[str, float]:
        return copy.deepcopy(self.repo)


class MicroSnippetSemanticCalulation:
    """
    Designed to calculate the semantic equality of two code snippets. The
    result indicates whether the two code snippets are semantically equal.

    For each snippet pair, it follows the following steps:
        1. collect the invocations in the snippets and compare them.
        2. collect all symbols in the snippets and init them with a
            random value.
        3. iterate the statements while updating the symbols' values.
            Also record the memory location's value meeting during the
            iteration.
        4. compare the symbol and memory location values to decide the
            equality.

    Q1: what about the if conditions and loop conditions?

    NOTE that currently we assume the code snippet is the function definition
    to satisfy the requirement of cinspector while generating the execution
    path.

    Attributes:
        code: the code snippet
        sym_table: the symbol table
        call_table: log the invocation-related information
    """

    def __init__(self, code: str, ref_sym_table: Optional[SymTable] = None, ref_call_table: Optional[Dict[str, float]] = None):
        """
        do some magic

        1. replace a + -1 with a - 1
        2. remove all cast
        """

        # 1
        t = re.findall(re.compile('\+ -\d'), code)
        if t:
            for _ in t:
                code = code.replace(_, _.replace('+ -', '- '))
        # 2
        cc = CCode(code)
        type_desc = cc.get_by_type_name('type_descriptor')
        tset = set()
        for desc in type_desc:
            if desc.parent.node_type == 'cast_expression':
                tset.add(f'({desc.src})')
        for _ in tset:
            code = code.replace(_, '')
        # print(f'code is \n\n {code} \n\n')
        """
        ==================== magic ends ===========================
        """

        self.code = code
        self.ref_sym_table = ref_sym_table
        ids = self._collect_symbols(code)
        self.sym_table = SymTable(ids)
        self.call_table = CallTable(ref_call_table)

        """
        used for ensuring two snippets have the same random initial values
        """
        if self.ref_sym_table:
            exact_key = set()
            similar_key = set()

            ref_dic_keys = self.ref_sym_table.dry().keys()
            symtab_keys = self.sym_table.dry().keys()
            # set exactly same key
            for k in ref_dic_keys:
                if k in symtab_keys:
                    self.sym_table.set_sym(k, self.ref_sym_table.get_sym(k))
                    exact_key.add(k)
            # set similar key
            for k in ref_dic_keys:
                similar_ids = [_ for _ in symtab_keys if distance(_, k) < 4 and _ not in exact_key]
                for si in similar_ids:
                    self.sym_table.set_sym(si, self.ref_sym_table.get_sym(k))
                    similar_key.add(si)

        # we need to backup the initial sym_table for different execution paths
        self.init_sym_table = copy.deepcopy(self.sym_table)

    def reset_symtable(self):
        self.sym_table = copy.deepcopy(self.init_sym_table)
        self.call_table = CallTable(self.call_table.reference_table)

    def _collect_symbols(self, code: str) -> Set[str]:
        """
        collect symbols in the code in order to send them
        to SymTable for initialization

        1. variable names
        2. call expression
        3. ? pointer expression, under consideration
        """

        cc = CCode(code)
        ids = cc.get_by_type_name('identifier')

        real_ids = set()
        for _id in ids:
            if _id.parent.node_type == 'call_expression':
                # _id = _id.parent
                continue
            elif _id.parent.node_type == 'pointer_expression':
                _id = _id.parent
            elif _id.parent.node_type == 'pointer_declarator':
                _id = _id.parent
            real_ids.add(_id.src)
        # collect pointer expression
        pointer_exp = cc.get_by_type_name('pointer_expression')
        for _ in pointer_exp:
            real_ids.add(_.src)
        return real_ids

    def test_collect_symbols(self):
        for _ in self._collect_symbols(self.code):
            print(_)

    def _each_path(self, path: List[Node]):
        """
        helper function of self.calculate, focus on emulating
        the execution of each path
        """

        for n in path:
            # print(f'* processing {n}')

            try:
                # process the (hidden) call expression like func(a, hidden(b))
                call_exps = n.descendants_by_type_name('call_expression')
                # we need to solve the nested first
                call_exps = Util.sort_nodes(call_exps, True)
                for exp in call_exps:
                    real_call = unpack_and_fill_call_expression(exp, self.sym_table, self.call_table)
                    if not self.call_table.has(real_call):
                        self.call_table.add(real_call)
            except Exception as e:
                print(e)
                pass

            if n.node_type == 'declaration':
                # process the init_declarator in the delcaration
                init_decl = n.descendants_by_type_name('init_declarator')
                if init_decl:
                    # int *a = malloc() -> a = malloc() - change to expression
                    exp = CCode(init_decl[0].src.lstrip('*') + ';').node.children[0]
                    self._each_path([exp])

            elif n.node_type == 'expression_statement':
                """
                Here we do magic to process the update_expression

                move this to the __init__ of MSSC later
                """
                n_src = n.src
                if n_src.endswith('++;'):
                    # print(f'n_src is {n_src}')
                    left_part = n_src[:n_src.index('++;')]
                    n_src = f"{left_part} = {left_part} + 1;"
                    n = CCode(n_src).node.children[0]
                    # print(f'new n_src is {n.src}')
                """
                magic ends
                """

                # a=1; is expression statement, a=1 is assignment expression
                if n.children[0].node_type == 'assignment_expression':
                    self._each_path([n.children[0]])
                elif n.children[0].node_type == 'call_expression':
                    self._each_path([n.children[0]])

            elif n.node_type == 'assignment_expression':
                self.process_assignment(n)

            elif n.node_type == 'init_declarator':
                self.process_init_declarator(n.declarator, n.value)

            elif n.node_type == 'call_expression':
                # here process the single call expression, expression like
                # a = b() will be processed in process_assignment or
                # process_init_declarator
                self.process_call_expression(n)
            else:
                # print(f'Undefined processing for {n}')
                pass

    def process_call_expression(self, n: CallExpressionNode):
        real_call = unpack_and_fill_call_expression(n, self.sym_table, self.call_table)
        self.call_table.add(real_call)

    def process_assignment(self, n):
        """
        decide how the assignment updates the symbol table

        1. whether left is in the table? if not, maybe a memory loc
        2. whether right is in the table? if not, maybe a memory loc
        """

        left = n.left
        right = n.right
        symbol = n.symbol

        # if right is a call_epxression
        # here we only add this call_expression to call_table
        # the assignment to left is conducted later
        if right.node_type == 'call_expression':
            real_call = unpack_and_fill_call_expression(right, self.sym_table, self.call_table)
            if not self.call_table.has(real_call):
                self.call_table.add(real_call)

        """
        first we need to unpack the assignment expression, e.g.,

            a += b -> a = a + b
        """

        exp = n.src + ';'
        if symbol == '|=':
            exp = f'{left.src} = {left.src} | {right.src};'
        if symbol == '~=':
            exp = f'{left.src} = {left.src} ~ {right.src};'
        if symbol == '+=':
            exp = f'{left.src} = {left.src} + {right.src};'
        if symbol == '-=':
            exp = f'{left.src} = {left.src} - {right.src};'
        if symbol == '*=':
            exp = f'{left.src} = {left.src} * {right.src};'
        if symbol == '/=':
            exp = f'{left.src} = {left.src} / {right.src};'

        exp = CCode(exp).get_by_type_name('assignment_expression')[0]
        left = exp.left
        right = exp.right

        try:
            right_value = self.conclude_right_value(right)
        except ValueError:
            # print(f'[process_assignment] Fail to process assignment {left.src} = {right.src}')
            return
        self.sym_table.set_sym(left.src, right_value)

    def conclude_right_value(self, right: BasicNode) -> Sym:
        """
        conclude the value of a right value. For example,
        a = b, c = *b + 1, b and *b + 1 belong to the right value.

        1. whether the instant number?
        2. whether a symbol in the symbol table?
        3. if it doesn't satisfy the 1 and 2, is it a memory location?
        4. unknown
        """

        def is_number(s: str) -> bool:
            # print(f'checking {repr(s)}')
            pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)$'
            match = re.match(pattern, s)
            res = match is not None
            # print(f'res is {res}')
            return match is not None

        def is_memory_loc(value: BasicNode) -> bool:
            # TODO, may be some heuristic rules
            return False

        def try_math_expression(value: BasicNode) -> Optional[float]:
            """
            whether a computable math expression

            Returns:
                None if not computable, otherwise the value
            """

            value_src = value.src

            # a. replace all the pointer expression in the right
            pointer_exps = value.descendants_by_type_name('pointer_expression')
            for _ in pointer_exps:
                sym = self.sym_table.get_sym(_.src)
                if sym:
                    value_src = value_src.replace(_.src, str(sym.get_value()))
                    # print(f'[replace pointer expression] replace {_.src} with {str(sym.get_value())}, new value_src: {value_src}')

            # a.1 replace all the call_expression in the right
            call_exps = CCode(value_src).get_by_type_name('call_expression')
            for _ in call_exps:
                real_call = unpack_and_fill_call_expression(_, self.sym_table, self.call_table)
                # print(f' we are trying to replace {real_call}')
                # print(f' call_table is {self.call_table.dry()}')
                if self.call_table.has(real_call):
                    val = self.call_table.get_value(real_call)
                    value_src = value_src.replace(_.src, str(val))
                    # print(f'replace {_.src} with {str(val)}, new src: {value_src}')

            # b. herustic rules
            # b.1 tree-sitter wrongly parse (ulong)(a+1) as the call
            value_src = value_src.replace('(ulong)', '')

            # c. all identifiers are in the symbol table with a concrete value
            ids = CCode(value_src).get_by_type_name('identifier')
            for _id in ids:
                if _id.parent.node_type == 'call_expression':
                    continue
                _id = _id.src
                sym = self.sym_table.get_sym(_id)
                if not sym or not is_number(str(sym.get_value())):
                    # print(f'not sym: {str(not sym)}')
                    is_n = is_number(str(sym))
                    # print(f'is number: {is_n}')
                    # print(f'detect unknown identifier {_id}')
                    return None
                value_src = value_src.replace(_id, str(sym))

            try:
                # print(f'eval - {value_src} ---------------')
                res = eval(value_src)
                return float(res)
            except Exception as e:
                # print(f'error in try_math_expression ' + str(e))
                return None

        if right.node_type == 'cast_expression':
            right = right.value

        if is_number(right.src):
            # number
            return Sym(right.src, SType.NUMBER)
        elif self.sym_table.get_sym(right.src):
            # exist in the symbol table
            return self.sym_table.get_sym(right.src)
        elif res := try_math_expression(right):
            if res:
                return Sym(res, SType.NUMBER)
        else:
            raise ValueError(f'unknown right value: {right.src}')

    def process_init_declarator(self, declarator: BasicNode, value: BasicNode):
        """
        decide how the init_declarator updates the symbol table

        conclude the <value>:
            1. is it the number?
            2. is it a symbol in symbol table?
            3. is it a memory location?
            4. unknown
        """

        try:
            right_value = self.conclude_right_value(value)
            self.sym_table.set_sym(declarator.src, right_value)
        except Exception as e:
            # print(e)
            # print(f'Fail to process the init_declarator {declarator}')
            pass

    def calculate(self, code_snippet: str = None) -> List[Dict[str, Union[str, int]]]:
        """
        calculate the symbol table of each path

        Returns:
            a list containing the symbol table of each path
        """

        # 1. generate the execution path of the snippet
        # currently we assume the snippet is the function definition
        if not code_snippet:
            code_snippet = self.code

        cc = CCode(code_snippet)
        func: FunctionDefinitionNode = cc.get_by_type_name('function_definition')[0]
        cfg = CFG(func)
        exe_paths: List = cfg.execution_path()

        # 2. iterate each path, calculate the symbol and memory location's value
        lst = []
        for path in exe_paths:
            # print('-'*10 + 'path' + '-'*10)
            # for n in path:
            #    print(n)
            self.reset_symtable()
            # self.sym_table.print_sym('before processing')
            self._each_path(path)
            # self.sym_table.print_sym('after processing')
            lst.append([self.sym_table.dry(), self.call_table.dry()])
        return lst


def mssc_testcase(name: str):
    cur = os.path.dirname(__file__)
    testcase = os.path.join(cur, 'mssc_testcase/' + name)
    with open(testcase, 'r') as r:
        testcase = r.read()
    cc = CCode(testcase)
    funcs = cc.get_by_type_name('function_definition')
    """
    func = funcs[0].src
    mssc = MicroSnippetSemanticCalulation(func)
    mssc.test_collect_symbols()
    exit()
    """
    sc = SemanticComparison(funcs[0].src, funcs[1].src)
    is_equal = (sc.is_semantic_equal())
    print(f'---------is_equal: {is_equal}--------------')
    print('\n'.join(sc.logger))


def single_code(code: str) -> None:
    """
    output the calculate of single code snippet
    """

    mssc = MicroSnippetSemanticCalulation(code)
    mssc.calculate()


if __name__ == '__main__':
    mssc_testcase('test0')
