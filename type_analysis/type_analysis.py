import tac
import tac_analysis
import program_cfg
import types_predicate_abstraction
from types_predicate_abstraction import ConcreteTransformer
import z3_utils
import z3

def test_code():
    x = 0
    z = x + 1
    z = x + 'bla'
    
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
            verbose=True):
# succ is the successor nodes in the CFG
# s in nodes is the start vertex
# i is the initial value at the start
# bottom is the minimum value
    wl = [s]
    df = dict() # this is df_enter (possible types before execution of the statement)
    df[s] = i
    while wl != []:
        u = wl.pop()
        if verbose: print("Handling:", u)
        
        for v in succ(u):
            if v not in df:
                df[v] = bottom
                
            new = join(df[v], abstract_transformer(u, v, df[u]))
            if verbose: print("changing the dataflow value at node:", v, "to" , new)
            if not abstract_equivalence(new, df[v]):
                df[v] = new
                wl.append(v)
            else:
                print("not changing")
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
    
def generate_type_safety_constraints(type_abstract_analyzer, cfg):
    return set(type_abstract_analyzer.generate_safety_constraints(program_location) 
               for program_location in cfg.all_locations())       

def check_type_safety(abstract_type_for_location, type_safety_constraints, types_exclusion_theory):
    solver = z3.Solver()
    for loc_abstract_type in abstract_type_for_location.values():
        solver.add(loc_abstract_type.formula())
        
    for exluction_statement in types_exclusion_theory:
        solver.add(exluction_statement)
        
    if z3_utils.unsat(solver):
        # this can happen with 3-address instructions that are not typable, e.g.
        # y = 'hello' + 1
        return False
        
    solver.add(z3.Not(z3.And(*type_safety_constraints)))
    return z3_utils.unsat(solver)

def analyze_type_safety(code_to_analyze):
    #ELAZAR: tried to change here so it will work with the updated API
    #I can't test it though :(
    cfg = tac_analysis.make_tacblock_cfg(code_to_analyze)
    tac_analysis.print_tac_cfg(cfg)

    cfg = program_cfg.ProgramCFG(cfg, name=tac_analysis.BLOCKNAME)
    cfg.depict()
    print(cfg.program_vars())
    
    type_abstract_analyzer = types_predicate_abstraction.AbstractTypesAnalysis(cfg)
    abstract_type_for_reachable_location = chaotic_type_analysis(type_abstract_analyzer, cfg)
    type_safety_constraints = generate_type_safety_constraints(type_abstract_analyzer, cfg)
    types_exclusion_theory = type_abstract_analyzer.get_types_exclusion_theory()
    print(type_safety_constraints)
    return check_type_safety(abstract_type_for_reachable_location, 
                         type_safety_constraints,
                         types_exclusion_theory)
    
def test():
    is_safe = analyze_type_safety(test_code)
    if is_safe:
        print("Type safe!")
    else:
        print("Not safe!")
            
if __name__ == '__main__':
    test()