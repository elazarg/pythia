import networkx as nx
import bcode
from bcode import UN_TO_OP


def accumulate_stack_depth(cfg):
    # edge_dfs() will give us edges to nodes we've already saw,
    # allowing us to validate that the stack is sensible on all paths
    cfg.node[-1]['depth_in'] = 0 
    for src, dst in nx.edge_dfs(cfg, -1):
        se = cfg.edge[src][dst]['stack_effect']
        dst_in_prev = cfg.node[dst].get('depth_in')
        dst_in = cfg.node[src]['depth_in'] + se
        cfg.node[dst]['depth_in'] = dst_in
        assert dst_in_prev in [None, dst_in]
        print(cfg.node[src]['ins'])
        assert dst_in >= 0


def make_graph(f):
    bs = bcode.get_instructions(f)
    dbs = dict((b.offset, b) for b in bs)
    cfg = nx.DiGraph([(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
                    for b in bs             # this ^ should be called weight to be used in algorithms
                    for (j, stack_effect) in b.next_list() if dbs.get(j)])
    for b in bs:
        cfg.node[b.offset]['ins'] = b
    accumulate_stack_depth(cfg)
    for b in bs:
        print(b.offset, ':', b.next_list(), cfg.node[b.offset].get('depth_in', 'DEAD CODE'), ' <- ', b)
    return cfg


def draw(g: nx.DiGraph):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g, with_labels=True)
    plt.show()

   
def test():
    import code_examples
    cfg = make_graph(code_examples.escapeval_to_color)
    print_3addr(cfg)
    #draw(cfg)


def to3addr(dep, ins, newdep, name):
    def var(x):
        return chr(ord('α') + x)
    if name == 'START':
        return 'START'
    elif name.startswith('POP_JUMP_IF_'):
        return 'IF {}{} GOTO {}'.format('' if name.endswith('TRUE') else 'NOT ', 
            var(dep), ins.argval)
    elif name.endswith('_VALUE'):
        return '{} {}'.format(name.split('_')[0], var(dep))
    elif name == 'JUMP_ABSOLUTE':
        return 'GOTO {}'.format(ins.argval)
    elif name == 'JUMP_FORWARD':
        return 'GOTO {}'.format(ins.argval)
    elif name == 'BUILD_SLICE':
        if ins.argval == 2:
            fmt = '{} = SLICE({}:{})'
        else:
            fmt = '{} = SLICE({}:{}:{})'
        return fmt.format(var(newdep), var(dep), var(dep-1), var(dep-2))
    elif name.startswith('BUILD_'):
        mid = ', '.join(var(i + 1) for i in range(dep - ins.argval, dep))
        return '{} = {}({})'.format(var(newdep), name.split('_')[-1], mid)
    elif name == 'CALL_FUNCTION':
        nargs = ins.argval & 0xFF
        nkwargs = 2*((ins.argval >> 8) & 0xFF)
        total = nargs + nkwargs
        mid = ', '.join(var(i + 1)
                    for i in range(dep - nkwargs - nargs, dep - nkwargs))
        mid_kw = ', '.join( (var(i) + ': ' + var(i+1))
                            for i in range(dep - nkwargs + 1, dep, 2))
        return '{} = {}({}, kw=({}))'.format(var(newdep), var(dep - total), mid, mid_kw)
    elif name == 'GET_ITER':
        return '{0} = iter({0})'.format(var(dep))
    elif name == 'FOR_ITER':
        return '{} = next({}) HANDLE: DEL {} and GOTO {}'.format(var(newdep), var(dep), var(dep), ins.argval)
    elif name.startswith('LOAD_FAST'):
        return '{} = {}'.format(var(newdep), ins.argval)
    elif name.startswith('LOAD_DEREF'):
        return '{} = NONLOCAL.{}'.format(var(newdep), ins.argval)
    elif name.startswith('LOAD_CONST'):
        return '{} = {}'.format(var(newdep), repr(ins.argval))
    elif name.startswith('LOAD_ATTR'):
        return '{} = {}.{}'.format(var(newdep), var(dep), ins.argval)
    elif name.startswith('LOAD_GLOBAL'):
        return '{} = GLOBALS.{}'.format(var(newdep), ins.argval)
    elif name == 'STORE_GLOBAL':
        return 'GLOBALS.{} = {}'.format(ins.argval, var(dep))
    elif name.startswith('STORE_ATTR'):
        return '{}.{} = {}'.format(var(dep), ins.argval, var(dep-1))
    elif name.startswith('STORE_SUBSCR'):
        return '{}[{}] = {}'.format(var(dep-1), var(dep), var(dep-2))
    elif name.startswith('STORE_FAST'):
        return '{} = {}'.format(ins.argval, var(dep))
    elif name == 'POP_TOP':
        return 'DEL {}'.format(var(dep))
    elif name == 'DELETE_FAST':
        return 'DEL {}'.format(ins.argval, var(dep))
    elif name == 'POP_BLOCK':
        return 'END LOOP'
    elif name == 'SETUP_LOOP':
        return 'LOOP'
    elif name == 'BREAK_LOOP':
        return 'BREAK {}'.format(ins.argrepr)
    elif 'LOOP' in name:
        return name.split('_')[0]
    elif name == 'COMPARE_OP':
        return '{} = {} {} {}'.format(var(newdep), var(dep), ins.argval, var(newdep))
    elif name.startswith('BINARY'):
        fmt = '{} = {} {} {}'
        if name == 'BINARY_SUBSCR':
            fmt = '{0} = {1}[{3}]'
        op = bcode.BIN_TO_OP[name.split('_', 1)[-1]]
        return fmt.format(var(newdep), var(dep - 1), op, var(dep))
    elif name.startswith('INPLACE'):
        fmt = '{} {}= {}'
        op = bcode.BIN_TO_OP[name.split('_')[-1]]
        return fmt.format(var(dep - 1), op, var(dep))
    elif ins.opcode <= 5:
        return '[STACK OP]'
    elif name.startswith('UNARY'):
        return '{} = {}{}'.format(var(newdep), UN_TO_OP[name], var(dep))
    elif name == 'UNPACK_SEQUENCE':
        seq = ', '.join(var(dep+i) for i in reversed(range(ins.argval)))
        return '{} = {}'.format(seq, var(dep))
    elif name == 'RAISE_VARARGS':
        return 'raise {}'.format(var(dep))
    elif name == 'IMPORT_NAME':
        return '{} = IMPORT {}'.format(var(newdep), ins.argval)
    elif name == 'IMPORT_FROM':
        return 'FROM {} IMPORT {}'.format(var(dep), ins.argval)
    else:
        print(ins)
        return '{} = {} {}'.format(var(newdep), ins.opname, var(dep))


def print_3addr(cfg):
    for n in sorted(cfg.node):
        d = cfg.node[n]
        ins = d['ins']
        dep = d.get('depth_in')
        if dep is None:
            continue
        newdep = dep + ins.stack_effect()
        name = ins.opname
        cmd = to3addr(dep, ins, newdep, name)
        print(n,':\t', cmd , '\t\t', '' and ins)

        
def is_inplace_unary(name):
    return name == 'GET_ITER'
      
if __name__ == '__main__':   
    test()
