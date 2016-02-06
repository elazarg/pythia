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
    cfg = nx.DiGraph()
    for b in bs:
        cfg.add_node(b.offset)
        cfg.node[b.offset]['ins'] = b
    dbs = dict((b.offset, b) for b in bs)
    cfg.add_edges_from([(b.offset, dbs[j].offset, {'stack_effect': stack_effect})
                    for b in bs             # this ^ should be called weight to be used in algorithms
                    for (j, stack_effect) in b.next_list() if dbs.get(j)])
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
    cfg = make_graph(code_examples.getpass)
    print_3addr(cfg)
    #draw(cfg)


def to3addr(ins, tos):
    new_tos = tos + ins.stack_effect()
    name = ins.opname
    val = ins.argval
    def var(x):
        return chr(ord('α') + x)
    if name == 'START':
        return 'START'
    if name == 'RAISED':
        return 'RAISED'
    elif name == 'RETURN_VALUE':
        return 'RETURN {}'.format(var(tos))
    elif name == 'YIELD_VALUE':
        return 'YIELD {}'.format(var(tos))
    elif name.startswith('POP_JUMP_IF_'):
        return 'IF {}{} GOTO {}'.format('' if name.endswith('TRUE') else 'NOT ', 
            var(tos), val)
    elif name == 'JUMP_ABSOLUTE':
        return 'GOTO {}'.format(val)
    elif name == 'JUMP_FORWARD':
        return 'GOTO {}'.format(val)
    elif name == 'BUILD_SLICE':
        if val == 2:
            fmt = '{0} = SLICE({2}:{1})'
        else:
            fmt = '{0} = SLICE({1}:{2}:{3})'
        return fmt.format(var(new_tos), var(tos), var(tos-1), var(tos-2))
    elif name.startswith('BUILD_'):
        mid = ', '.join(var(i + 1) for i in range(tos - val, tos))
        return '{} = {}({})'.format(var(new_tos), name.split('_')[-1], mid)
    elif name == 'GET_ITER':
        return '{0} = iter({0})'.format(var(tos))
    elif name == 'FOR_ITER':
        return '{} = next({}) HANDLE: DEL {} and GOTO {}'.format(var(new_tos), var(tos), var(tos), val)
    elif name.startswith('LOAD_FAST'):
        return '{} = {}'.format(var(new_tos), val)
    elif name.startswith('LOAD_DEREF'):
        return '{} = NONLOCAL.{}'.format(var(new_tos), val)
    elif name.startswith('LOAD_CONST'):
        return '{} = {}'.format(var(new_tos), repr(val))
    elif name.startswith('LOAD_ATTR'):
        return '{} = {}.{}'.format(var(new_tos), var(tos), val)
    elif name.startswith('LOAD_GLOBAL'):
        return '{} = GLOBALS.{}'.format(var(new_tos), val)
    elif name == 'STORE_GLOBAL':
        return 'GLOBALS.{} = {}'.format(val, var(tos))
    elif name.startswith('STORE_ATTR'):
        return '{}.{} = {}'.format(var(tos), val, var(tos-1))
    elif name.startswith('STORE_SUBSCR'):
        return '{}[{}] = {}'.format(var(tos-1), var(tos), var(tos-2))
    elif name.startswith('STORE_FAST'):
        return '{} = {}'.format(val, var(tos))
    elif name == 'POP_TOP':
        return 'DEL {}'.format(var(tos))
    elif name == 'DELETE_FAST':
        return 'DEL {}'.format(val, var(tos))
    elif name == 'POP_BLOCK':
        return 'END LOOP'
    elif name == 'SETUP_LOOP':
        return 'LOOP'
    elif name == 'BREAK_LOOP':
        return 'BREAK {}'.format(ins.argrepr)
    elif name == 'CONTINUE_LOOP':
        return 'BREAK {}'.format(ins.argrepr)
    elif name == 'RAISE_VARARGS':
        return 'raise {}'.format(var(tos))
    elif name == 'COMPARE_OP':
        return '{} = {} {} {}'.format(var(new_tos), var(tos), val, var(new_tos))
    elif name.startswith('UNARY'):
        return '{} = {}{}'.format(var(new_tos), UN_TO_OP[name], var(tos))
    elif name == 'BINARY_SUBSCR':
        return '{0} = {1}[{2}]'.format(var(new_tos), var(tos - 1), var(tos))
    elif name.startswith('BINARY'):
        op = bcode.BIN_TO_OP[name.split('_', 1)[-1]]
        return '{} = {} {} {}'.format(var(new_tos), var(tos - 1), op, var(tos))
    elif name.startswith('INPLACE_'):
        fmt = '{} {}= {}'
        op = bcode.BIN_TO_OP[name.split('_')[-1]]
        return fmt.format(var(tos - 1), op, var(tos))
    elif ins.opcode <= 5:
        return '[STACK OP]'
    elif name == 'UNPACK_SEQUENCE':
        seq = ', '.join(var(tos+i) for i in reversed(range(val)))
        return '{} = {}'.format(seq, var(tos))
    elif name == 'IMPORT_NAME':
        return '{} = IMPORT {}'.format(var(new_tos), val)
    elif name == 'IMPORT_FROM':
        return '{} = FROM {} IMPORT {}'.format(var(new_tos), var(tos), val)
    elif name == 'CALL_FUNCTION':
        nargs = val & 0xFF
        nkwargs = 2*((val >> 8) & 0xFF)
        total = nargs + nkwargs
        mid = ', '.join(var(i + 1)
                    for i in range(tos - nkwargs - nargs, tos - nkwargs))
        mid_kw = ', '.join( (var(i) + ': ' + var(i+1))
                            for i in range(tos - nkwargs + 1, tos, 2))
        return '{} = {}({}, kw=({}))'.format(var(new_tos), var(tos - total), mid, mid_kw)
    else:
        print(ins)
        return '{} = {} {}'.format(var(new_tos), ins.opname, var(tos))


def print_3addr(cfg):
    for n in sorted(cfg.node):
        d = cfg.node[n]
        ins = d['ins']
        tos = d.get('depth_in')
        if tos is None:
            # unreachable code
            continue
        cmd = to3addr(ins, tos)
        print(n,':\t', cmd , '\t\t', '' and ins)

        
def is_inplace_unary(name):
    return name == 'GET_ITER'
      
if __name__ == '__main__':   
    test()
