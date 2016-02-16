import tac
import tac_analysis
import program_cfg

def test_code():
    while True:
        x = 1
    y = x + 2
    

def chaotic(succ, s, i, join, bottom, tr, tr_txt):
# succ is the successor nodes in the CFG
# s in nodes is the start vertex
# i is the initial value at the start
# bottom is the minimum value
# tr is the transfer function
# tr_txt is the edge annotations
    wl = [s]
    df = dict([(x, bottom) for x in succ])
    df[s] = i
    while wl != []:
        u = wl.pop()
        print("Handling:", u)
        for v in succ[u]:
            new = join(tr[(u,v)](df[u]), df[v])
            if (new != df[v]):
                print("changing the dataflow value at node:", v, "to" , new)
                df[v] = new
                wl.append(v)
                wl.sort(key=lambda x:-x) # sort in backward key order

    
    print("Dataflow results")
    for node in succ:
        print("%s: %s"%(node, df[node]))
    import os
    f = open("temp_chaotic.dt", "w")
    f.write("digraph cfg {\n")
    # write nodes and df values
    for node in succ:
        f.write("\t" + str(node) + " [label=\"df=" + str(df[node]) + "\"]\n")
        
    for u in succ:
        for v in succ[u]:
            f.write("\t" + str(u) + "->" + str(v) + " [label=\"" + tr_txt[(u,v)]+"\"]\n")
    f.write("\t}\n")
    f.close()
    os.system("dot temp_chaotic.dt -Tpng > chaotic.png")
    os.system("chaotic.png")       
    
def test():
    name = "what is this good for"
    cfg = tac.make_tacblock_cfg(test_code, blockname=name)
    cfg = tac_analysis.simplified_tac_blocks_cfg(cfg, blockname=name)
    print(cfg.nodes())
    assert False
    #tac_analysis.print_tac_cfg(cfg, blockname=name)

    cfg = program_cfg.ProgramCFG(cfg, name)
    cfg.depict()
    
        
if __name__ == '__main__':
    test()