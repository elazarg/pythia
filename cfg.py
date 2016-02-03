import networkx as nx
import bcode


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
    cfg = make_graph(code_examples.example)
    print_3addr(cfg)
    #draw(cfg)

def print_3addr(cfg):
    def var(x):
        return '@' + chr(ord('a') + x)
    for n in sorted(cfg.node):
        d = cfg.node[n]
        dep = d['depth_in'] - 1
        ins = d['ins']
        newdep = dep + ins.stack_effect()
        name = ins.opname
        print(n,':', end='\t')
        if name == 'START':
            cmd = 'START'
        elif name == 'POP_TOP':
            cmd = 'DEL {}'.format(var(dep))
        elif name.startswith('POP_JUMP_IF_'):
            cmd = 'IF {}{} GOTO {}'.format('' if name.endswith('TRUE') else 'NOT ',
                                           var(dep), ins.argval)
        elif name.endswith('_VALUE'):
            cmd = '{} {}'.format(name.split('_')[0], var(dep))
        elif name == 'JUMP_ABSOLUTE':
            cmd = 'GOTO {}'.format(ins.argval)
        elif name == 'JUMP_FORWARD':
            cmd = 'GOTO {}'.format(ins.argval)
        elif name.startswith('BUILD_'):
            mid = ', '.join(var(i+1) for i in range(dep-ins.argval, dep))
            cmd = '{} = {}({})'.format(var(newdep), name.split('_')[-1], mid)
        elif name == 'CALL_FUNCTION':
            mid = ', '.join(var(i+1) for i in range(dep-ins.argval, dep))
            cmd = '{} = {}({})'.format(var(newdep), var(dep-ins.argval), mid)
        elif name == 'GET_ITER':
            cmd = '{0} = iter({0})'.format(var(dep))
        elif name == 'FOR_ITER':
            cmd = '{} = next({}) HANDLE: DEL {} and GOTO {}'.format(var(newdep), var(dep), var(dep), ins.argval)
        elif name.startswith('LOAD_FAST'):
            cmd = '{} = {}'.format(var(newdep), ins.argval)
        elif name.startswith('LOAD_DEREF'):
            cmd = '{} = NONLOCAL.{}'.format(var(newdep), ins.argval)
        elif name.startswith('LOAD_CONST'):
            cmd = '{} = {}'.format(var(newdep), repr(ins.argval))
        elif name.startswith('LOAD_ATTR'):
            cmd = '{} = {}.{}'.format(var(newdep), var(dep), ins.argval)
        elif name.startswith('LOAD_GLOBAL'):
            cmd = '{} = GLOBALS.{}'.format(var(newdep), ins.argval)
        elif name.startswith('STORE_'):
            cmd = '{} = {}'.format(ins.argval, var(dep))
        elif name == 'POP_BLOCK':
            cmd = 'END LOOP'
        elif name == 'SETUP_LOOP':
            cmd = 'LOOP'
        elif 'LOOP' in name:
            cmd = name.split('_')[0]
        elif name == 'COMPARE_OP':
            cmd = '{} = {} {} {}'.format(var(newdep), var(dep), ins.argval, var(newdep))
        elif name.startswith('BINARY'):
            fmt = '{} = {} {} {}'
            if name == 'BINARY_SUBSCR':
                fmt = '{0} = {1}[{3}]'
            cmd = fmt.format(var(newdep), var(dep-1), bcode.BIN_TO_OP[name], var(dep))
        else:
            cmd = '{} = {} {}'.format(var(newdep), ins.opname, var(dep))
        print(cmd , '\t\t', ins)
        
def is_inplace_unary(name):
    return name == 'GET_ITER'
      
if __name__ == '__main__':   
    test()
