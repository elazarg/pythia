import tac
import tac_analysis
import program_cfg
import types_predicate_abstraction
from types_predicate_abstraction import ConcreteTransformer
import z3

# def test_code():
#     while True:
#         w = "bla"
#         v = 1
#     z = w + v
def test_code():
    y = 0
    while True:
        y = 1
    z = 1 + y
    
def chaotic_type_analysis(type_abstract_analyzer, cfg):
    s = cfg.initial_location()
    i = types_predicate_abstraction.AbstractType(True)
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
            join, bottom,
            verbose=False):
# succ is the successor nodes in the CFG
# s in nodes is the start vertex
# i is the initial value at the start
# bottom is the minimum value
    wl = [s]
    df = dict()
    df[s] = i
    while wl != []:
        u = wl.pop()
        if verbose: print("Handling:", u)
        
        for v in succ(u):
            if v not in df:
                df[v] = bottom
                
            new = join(df[v], abstract_transformer(u, v, df[u]))
            if not abstract_equivalence(new, df[v]):
                if verbose: print("changing the dataflow value at node:", v, "to" , new)
                df[v] = new
                wl.append(v)
                # TODO: use a good chaotic ordering
                # Shachar's implementation has here:
                # wl.sort(key=lambda x:-x) # sort in backward key order

    print("Dataflow results")
    for node in all_nodes:
        if node not in df:
            print("%s: unreachable" % str(node))
        else:
            print("%s: %s"%(node, df[node]))
    return df

#     import os
#     f = open("temp_chaotic.dt", "w")
#     f.write("digraph cfg {\n")
#     # write nodes and df values
#     for node in succ:
#         f.write("\t" + str(node) + " [label=\"df=" + str(df[node]) + "\"]\n")
#         
#     for u in succ:
#         for v in succ[u]:
#             f.write("\t" + str(u) + "->" + str(v) + " [label=\"" + tr_txt[(u,v)]+"\"]\n")
#     f.write("\t}\n")
#     f.close()
#     os.system("dot temp_chaotic.dt -Tpng > chaotic.png")
#     os.system("chaotic.png")
    
def generate_type_safety_constraints(type_abstract_analyzer, cfg):
    return set(type_abstract_analyzer.generate_safety_constraints(program_location) 
               for program_location in cfg.all_locations())       

def check_type_safety(abstract_type_for_location, type_safety_constraints):
    solver = z3.Solver()
    for loc_abstract_type in abstract_type_for_location.values():
        solver.add(loc_abstract_type.formula())
    res = solver.check()
    assert res == z3.sat or z3.unsat, res
    if res == z3.unsat:
        return False
    
    solver.add(z3.Not(z3.And(*type_safety_constraints)))
    is_sat = solver.check()
    assert is_sat == z3.sat or is_sat == z3.unsat
    return is_sat == z3.unsat
    
def test():
    name = "bcode_block"
    cfg = tac.make_tacblock_cfg(test_code, blockname=name)
    cfg = tac_analysis.simplified_tac_blocks_cfg(cfg, blockname=name)
    tac_analysis.print_tac_cfg(cfg, blockname=name)

    cfg = program_cfg.ProgramCFG(cfg, name)
    cfg.depict()
    print(cfg.program_vars())
    
    type_abstract_analyzer = types_predicate_abstraction.AbstractTypesAnalysis(cfg)
    abstract_type_for_reachable_location = chaotic_type_analysis(type_abstract_analyzer, cfg)
    type_safety_constraints = generate_type_safety_constraints(type_abstract_analyzer, cfg)
    print(type_safety_constraints)
    if check_type_safety(abstract_type_for_reachable_location, 
                         type_safety_constraints):
        print("Type safe!")
    else:
        print("Not safe!")
        
if __name__ == '__main__':
    test()