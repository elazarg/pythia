import tac
import tac_analysis

def test_function():
    x = 1
    if x:
        y = 1
    else:
        x = 1


def inst_to_ivy(inst):
    return inst.fmt.format(**inst._asdict()) + '\n'
#     if :
#         return """
#         """.format(,,,)
#     elif :
#         return """
#         """.format(,,,)
#     elif :
#         return """
#         """.format(,,,)
#     elif :
#         return """
#         """.format(,,,)
#     elif :
#         return """
#         """.format(,,,)
#     elif :
#         return """
#         """.format(,,,)
#     else:
#         assert False, inst

def tac_blocks_cfg_to_ivy(tac_blocks_cfg):
    st = """
    relation str(X)
    relation num(X)
    relation bool(X)
    function object -> type
    
    
    """
    first_block_id = min(tac_blocks_cfg.nodes())
    for node_id in tac_blocks_cfg.nodes():
        block_instructions = tac_blocks_cfg.node[node_id][tac.BLOCKNAME]
        successor_ids = tac_blocks_cfg.successors(node_id)
        

        st += "action {} = {{\n".format(node_id)
        for inst in block_instructions:
            st += inst_to_ivy(inst)
        st += "}\n"
    return st
        
def test(code_to_analyze):
    tac_blocks_cfg = tac_analysis.make_tacblock_cfg(code_to_analyze)
    tac_analysis.print_tac_cfg(tac_blocks_cfg)
    print()
    print(tac_blocks_cfg_to_ivy(tac_blocks_cfg))
    
if __name__ == "__main__":
    test(test_function)