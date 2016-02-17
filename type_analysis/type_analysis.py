import tac
import tac_analysis
import program_cfg
import types_predicate_abstraction
from types_predicate_abstraction import ConcreteTransformer

def test_code():
    while True:
        x = 1
    y = x + 2
    
def chaotic_type_analysis(type_abstract_analyzer, cfg):
    s = cfg.initial_location()
    i = types_predicate_abstraction.AbstractType()
    all_nodes = cfg.all_locations()
    succ = lambda location: cfg.successors(location)
    abstract_transformer = lambda loc1, loc2, abstract_type: \
                                type_abstract_analyzer.transform(ConcreteTransformer(loc1, loc2),
                                                                 abstract_type)
    abstract_equivalence = lambda abs_type1, abs_type2: type_abstract_analyzer.equiv_type(abs_type1, abs_type2)
    join = lambda abs_type1, abs_type2: type_abstract_analyzer.join(abs_type1, abs_type2)
    bottom = type_abstract_analyzer.lattice_bottom()
    return chaotic(s, i, 
                all_nodes, succ,
                abstract_transformer, abstract_equivalence, 
                join, bottom)

# Credit to Shachar Itzhaky
def chaotic(s, i, 
            all_nodes, succ,
            abstract_transformer, 
            abstract_equivalence,
            join, bottom):
# succ is the successor nodes in the CFG
# s in nodes is the start vertex
# i is the initial value at the start
# bottom is the minimum value
    wl = [s]
    df = dict([(x, bottom) for x in all_nodes])
    df[s] = i
    while wl != []:
        u = wl.pop()
        print("Handling:", u)
        for v in succ(u):
            new = join(df[v], abstract_transformer(u, v, df[u]))
            if not abstract_equivalence(new, df[v]):
                print("changing the dataflow value at node:", v, "to" , new)
                df[v] = new
                wl.append(v)
                # TODO: use a good chaotic ordering
                # Shachar's implementation has here:
                # wl.sort(key=lambda x:-x) # sort in backward key order

    
    print("Dataflow results")
    for node in all_nodes:
        print("%s: %s"%(node, df[node]))
    return df

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
    name = "bcode_block"
    cfg = tac.make_tacblock_cfg(test_code, blockname=name)
    cfg = tac_analysis.simplified_tac_blocks_cfg(cfg, blockname=name)
    #tac_analysis.print_tac_cfg(cfg, blockname=name)

    cfg = program_cfg.ProgramCFG(cfg, name)
    cfg.depict()
    print(cfg.program_vars())
    
    type_abstract_analyzer = types_predicate_abstraction.AbstractTypesAnalysis(cfg)
    chaotic_type_analysis(type_abstract_analyzer, cfg)
    
        
if __name__ == '__main__':
    test()