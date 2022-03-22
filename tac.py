import itertools as it
from dataclasses import dataclass
from typing import NamedTuple, Iterable, Optional

import bcode
import bcode_cfg
import graph_utils as gu


def test() -> None:
    import code_examples
    cfg = make_tacblock_cfg(code_examples.calc_mandelbrot_vals)
    print_3addr(cfg)
    bcode_cfg.draw(cfg)


def linearize_cfg(cfg, no_dels=True) -> Iterable[str]:
    for n in sorted(cfg.nodes()):
        # The call for sorted() gives us for free the ordering of blocks by "fallthrough"
        # It is not guaranteed anywhere, but it makes sense - the order of blocks is the 
        # order of the lines in the code
        block = cfg.nodes[n]['block']
        for ins in block:
            if no_dels and isinstance(ins, Del):
                continue
            yield f'{n}:\t{ins}'


def print_3addr(cfg, no_dels=True) -> None:
    print('\n'.join(linearize_cfg(cfg, no_dels)))


def make_tacblock_cfg(f, simplify=False):
    depths, cfg = bcode_cfg.make_bcode_block_cfg_from_function(f)

    if simplify:
        cfg = gu.simplify_cfg(cfg)

    def bcode_block_to_tac_block(n, block_data: dict[str, list[bcode.BCode]]) -> list[Tac]:
        return list(it.chain.from_iterable(
            make_TAC(bc.opname, bc.argval, bc.stack_effect(), depths[bc.offset], bc.starts_line)
            for bc in block_data['block']))

    tac_blocks = gu.node_data_map(cfg,
                                  bcode_block_to_tac_block,
                                  'block')

    # this is simplistic kills analysis, that is not correct in general:
    # tac = [Del(tuple(get_gens(tac) - get_uses(tac))] + tac
    # if stack_effect < 0:
    #     tac += [Del(stackvar(stack_depth + i)) for i in range(stack_effect + 1, 1)]
    return tac_blocks


class Const(NamedTuple):
    value: object

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)


class Var(NamedTuple):
    name: str
    is_stackvar: bool = False

    def __str__(self):
        if self.is_stackvar:
            return f'${self.name}'
        return self.name

    def __repr__(self):
        if self.is_stackvar:
            return f'${self.name}'
        return self.name


class Attribute(NamedTuple):
    var: Var
    attr: str

    def __str__(self):
        return f'{self.var}.{self.attr}'


def stackvar(x) -> Var:
    return Var(str(x - 1), True)


def is_stackvar(v: Var | Const) -> bool:
    return isinstance(v, Var) and v.is_stackvar


# Simplified version of the real binding construct in Python.
Signature = Var | tuple[Var]

Value = Var | Attribute | Const


@dataclass
class Nop:
    pass


@dataclass
class Assign:
    target: Signature
    value: Value
    is_id: bool = True

    def __str__(self):
        return f'{self.target} = {self.value}'


@dataclass
class Import:
    lhs: Var
    modname: str
    feature: str = None

    def __str__(self):
        res = f'{self.lhs} = IMPORT {self.modname}'
        if self.feature is not None:
            res += f'.{self.feature}'
        return res


@dataclass
class Binary:
    lhs: Signature
    left: Value
    op: str
    right: Value
    is_id: bool = True
    is_inplace: bool = False

    def __str__(self):
        if self.is_inplace:
            return f'{self.lhs} {self.op}= {self.right}'
        return f'{self.lhs} = {self.left} {self.op} {self.right}'


@dataclass
class Call:
    lhs: Signature
    function: Var
    args: tuple[Value, ...]
    kwargs: Var = None

    def __str__(self):
        res = f'{self.lhs} = {self.function}{self.args}'
        if self.kwargs:
            res += f', kwargs={self.kwargs}'
        return res


@dataclass
class Jump:
    jump_target: str
    cond: Value = True

    def __str__(self):
        return f'IF {self.cond} GOTO {self.jump_target}'


@dataclass
class For:
    lhs: Signature
    iterator: Value
    jump_target: str

    def __str__(self):
        return f'{self.lhs} = next({self.iterator}) HANDLE: GOTO {self.jump_target}'


@dataclass
class Return:
    value: Value

    def __str__(self):
        return f'RETURN {self.value}'


@dataclass
class Raise:
    value: Value

    def __str__(self):
        return f'RAISE {self.value}'


@dataclass
class Yield:
    lhs: Signature
    value: Value


@dataclass
class Del:
    variables: tuple[Var]

    def __str__(self):
        return f'DEL {self.variables}'


@dataclass
class Unsupported:
    name: str


Tac = Nop | Assign | Import | Binary | Call | Jump | For | Return | Raise | Yield | Del | Unsupported

NOP = Nop()


def unary(lhs: Var, op) -> Tac:
    return call(lhs, op, args=(lhs,))


def assign_attr(lhs: Var, rhs: Var, attr, is_id=True) -> Assign:
    return Assign(lhs, Attribute(rhs, attr))


def mulassign(*lhs: Var, rhs: Var, is_id=True) -> Tac:
    return Assign(lhs, rhs, is_id=is_id)


def call(lhs: Var, f, args=(), kwargs: Optional[Var] = None) -> Tac:
    return Call(lhs, f, args, kwargs)


def foreach(lhs: Var, iterator, target) -> Tac:
    return For(lhs, iterator, target)


def jump(target, cond: Var | Const = Const(True)) -> Jump:
    return Jump(target, cond)


def get_gens(block) -> set[Var]:
    return set(it.chain.from_iterable(ins.gens for ins in block))


def get_uses(block) -> set[Var]:
    return set(it.chain.from_iterable(ins.uses for ins in block))


def make_TAC(opname, val, stack_effect, stack_depth, starts_line=None) -> list[Tac]:
    if opname == 'LOAD_CONST' and isinstance(val, tuple):
        lst = []
        for v in val:
            lst += make_TAC('LOAD_CONST', v, 1, stack_depth, starts_line)
            stack_depth += 1
        return lst + make_TAC('BUILD_TUPLE', len(val), -len(val) + 1, stack_depth, starts_line)
    tac = make_TAC_no_dels(opname, val, stack_effect, stack_depth)
    return tac  # [t._replace(starts_line=starts_line) for t in tac]


def make_TAC_no_dels(opname, val, stack_effect, stack_depth) -> list[Tac]:
    """Translate a bytecode operation into a list of TAC instructions.
    """
    out = stack_depth + stack_effect if stack_depth is not None else None
    name, op = choose_category(opname, val)
    match name:
        case 'UNARY_ATTR':
            return [unary(stackvar(stack_depth), op)]
        case 'UNARY_FUNC':
            return [call(stackvar(out), op, (stackvar(stack_depth),))]
        case 'BINARY':
            lhs = stackvar(out)
            left = stackvar(stack_depth - 1)
            right = stackvar(stack_depth)
            return [Binary(lhs, left, op, right, is_id=False, is_inplace=False)]
        case 'INPLACE':
            lhs1 = stackvar(out)
            rhs1 = stackvar(stack_depth)
            return [Binary(lhs1, lhs1, op, rhs1, is_id=False, is_inplace=True)]
        case 'POP_JUMP_IF_FALSE' | 'POP_JUMP_IF_TRUE' | 'POP_JUMP_IF_NONE' | 'POP_JUMP_IF_NOT_NONE':
            res = [call(stackvar(stack_depth), 'not', (stackvar(stack_depth),))] if name.endswith('FALSE') else []
            return res + [Jump(val, stackvar(stack_depth))]
        case 'JUMP':
            return [Jump(val)]
        case 'POP_TOP':
            return []
        case 'DELETE_FAST':
            return []
        case 'ROT_TWO':
            fresh = stackvar(stack_depth + 1)
            return [Assign(fresh, stackvar(stack_depth)),
                    Assign(stackvar(stack_depth), stackvar(stack_depth - 1)),
                    Assign(stackvar(stack_depth - 1), fresh),
                    Del(fresh)]
        case 'ROT_THREE':
            fresh = stackvar(stack_depth + 1)
            return [Assign(fresh, stackvar(stack_depth - 2)),
                    Assign(stackvar(stack_depth - 2), stackvar(stack_depth - 1)),
                    Assign(stackvar(stack_depth - 1), stackvar(stack_depth)),
                    Assign(stackvar(stack_depth), fresh),
                    Del(fresh)]
        case 'DUP_TOP':
            return [Assign(stackvar(out), stackvar(stack_depth))]
        case 'DUP_TOP_TWO':
            return [Assign(stackvar(out), stackvar(stack_depth - 2)),
                    Assign(stackvar(out + 1), stackvar(stack_depth - 1))]
        case 'RETURN_VALUE':
            return [Return(stackvar(stack_depth))]
        case 'YIELD_VALUE':
            return [Yield(stackvar(out), stackvar(stack_depth))]
        case 'FOR_ITER':
            return [foreach(stackvar(out), stackvar(stack_depth), val)]
        case 'LOAD':
            match op:
                case 'ATTR':
                    return [Call(stackvar(out), Var('BUILTINS.getattr'), (stackvar(stack_depth), Var(repr(val))))]
                case 'METHOD':
                    return [Call(stackvar(out), Var('BUILTINS.getattr'), (stackvar(stack_depth), Var(repr(val))))]
                case 'FAST' | 'NAME':
                    rhs = val
                case 'CONST':
                    rhs = Const(val)
                case 'DEREF':
                    rhs = Var('NONLOCAL.{}'.format(val))
                case 'GLOBAL':
                    rhs = Var('GLOBALS.{}'.format(val))
                case _:
                    assert False, op
            return [Assign(stackvar(out), rhs, is_id=(op != 'CONST'))]
        case 'STORE_FAST' | 'STORE_NAME':
            return [Assign(val, stackvar(stack_depth))]
        case 'STORE_GLOBAL':
            return [Assign(Var('GLOBALS.{}'.format(val)), stackvar(stack_depth))]
        case 'STORE_ATTR':
            return [Call(stackvar(stack_depth), Var('setattr'), Attribute(stackvar(stack_depth), val),
                         stackvar(stack_depth - 1))]
        case 'STORE_SUBSCR':
            return [Call(stackvar(stack_depth), Var('BUILTINS.getattr'), (stackvar(stack_depth - 1), Var('__setitem__'))),
                    Call(stackvar(stack_depth), stackvar(stack_depth), (stackvar(stack_depth - 2),))]
        case 'BINARY_SUBSCR':
            #
            # return [call(stackvar(out), 'BUILTINS.getattr', (stackvar(stack_depth - 1), "'__getitem__'")),
            #        call(stackvar(out), stackvar(out), (stackvar(stack_depth),))]
            # IVY-Specific: :(
            return [call(stackvar(out), 'BUILTINS.getitem', (stackvar(stack_depth - 1), stackvar(stack_depth)))]
        case 'POP_BLOCK':
            return [NOP]
        case 'SETUP_LOOP':
            return [NOP]
        case 'RAISE_VARARGS':
            return [Raise(stackvar(stack_depth))]
        case 'UNPACK_SEQUENCE':
            seq = tuple(stackvar(stack_depth + i) for i in reversed(range(val)))
            return [Assign(seq, stackvar(stack_depth))]
        case 'IMPORT_NAME':
            return [Import(stackvar(out), val)]
        case 'IMPORT_FROM':
            return [Import(stackvar(out), stackvar(stack_depth), val)]
        case 'BUILD':
            if op == 'SLICE':
                if val == 2:
                    args = (stackvar(stack_depth - 1), stackvar(stack_depth))
                else:
                    args = (stackvar(stack_depth), stackvar(stack_depth - 1), stackvar(stack_depth - 2))
                return [call(stackvar(out), 'SLICE', args)]
            return [call(stackvar(out), op, tuple(stackvar(i + 1) for i in range(stack_depth - val, stack_depth)))]
        case 'CALL_FUNCTION':
            nargs = val & 0xFF
            mid = [stackvar(i + 1) for i in range(stack_depth - nargs, stack_depth)]
            return [call(stackvar(out), stackvar(stack_depth - nargs), tuple(mid))]
        case 'CALL_FUNCTION_KW':
            nargs = val
            mid = [stackvar(i + 1) for i in range(stack_depth - nargs - 1, stack_depth - 1)]
            res = [Call(stackvar(out), stackvar(stack_depth - nargs - 1), tuple(mid), stackvar(stack_depth))]
            return res
        case "NOP":
            return []
    return [Unsupported(name)]


def choose_category(opname, argval) -> tuple[str, Optional[str]]:
    # NB: I'm not sure that this function is actually helpful.
    if opname in ('UNARY_POSITIVE', 'UNARY_NEGATIVE', 'UNARY_INVERT'):
        return 'UNARY_ATTR', UN_TO_OP[opname]
    if opname in ('GET_ITER', 'UNARY_NOT', 'GET_YIELD_FROM_ITER'):
        return 'UNARY_FUNC', UN_TO_OP[opname]

    if opname in ('JUMP_ABSOLUTE', 'JUMP_FORWARD', 'BREAK_LOOP', 'CONTINUE_LOOP'):
        return 'JUMP', None

    if opname in ("NOP",):
        return 'NOP', None

    desc = opname.split('_', 1)[1]
    if opname == 'COMPARE_OP':
        return 'BINARY', argval
    if opname.startswith('BINARY_') and opname != 'BINARY_SUBSCR':
        return 'BINARY', BIN_TO_OP[desc]

    if opname.startswith('INPLACE_'):
        return 'INPLACE', BIN_TO_OP[desc]

    if opname.startswith('BUILD_'):
        return 'BUILD', desc

    if opname.startswith('LOAD_'):
        return 'LOAD', desc

    return opname, None


BIN_TO_OP = {
    'POWER': '**',
    'MULTIPLY': '*',
    'MATRIX_MULTIPLY': '@',
    'FLOOR_DIVIDE': '//',
    'TRUE_DIVIDE': '/',
    'MODULO': '%',
    'ADD': '+',
    'SUBTRACT': '-',
    'SUBSCR': '[]',
    'LSHIFT': '<<',
    'RSHIFT': '>>',
    'AND': '&',
    'XOR': '^',
    'OR': '|'
}

# TODO. == to __eq__, etc.
CMPOP_TO_OP = {

}

UN_TO_OP = {
    # __abs__ ?
    'UNARY_POSITIVE': '__pos__',
    'UNARY_NEGATIVE': '__neg__',
    'UNARY_NOT': 'not ',
    'UNARY_INVERT': '__invert__',
    'GET_ITER': 'ITER ',
    'GET_YIELD_FROM_ITER': 'YIELD_FROM_ITER '
}

if __name__ == '__main__':
    test()
