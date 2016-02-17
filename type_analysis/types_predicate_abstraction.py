import tac
import z3

class ConcreteTransformer(object):
    def __init__(self, program_location_transforming, program_location_next):
        self._transforming = program_location_transforming
        self._next = program_location_next
        
    def current_location(self):
        return self._transforming
    
    def next_location(self):
        return self._next
    
class AbstractType(object):
    def __init__(self, formula=False):
        super(AbstractType, self).__init__()
        
        self._formula = formula
        #self._simplify()
        
    def formula(self):
        return self._formula
    
    def join_with(self, abstract_type):
        self._formula = z3.Or(self._formula, abstract_type._formula)
        self._simplify()
        
    def _simplify(self):
        self._formula = z3.simplify(self._formula)
        
    def __str__(self):
        return "AbstractType(%s)" % str(self._formula)

class AbstractTypesAnalysis(object):
    def __init__(self, cfg):
        super(AbstractTypesAnalysis, self).__init__()
        
        self._cfg = cfg
                
        self.TypeSort = z3.DeclareSort("Type")
        self._possible_type_relations = [z3.Function('numeric', self.TypeSort, z3.BoolSort()),
                                        z3.Function('string', self.TypeSort, z3.BoolSort())]
            
    def lattice_bottom(self):
        return AbstractType(formula=False)
        
    def _type_propagated_formula(self, previou_var_name, current_var_name):
        previous_var_logical_constant = z3.Const(previou_var_name, self.TypeSort)
        current_var_logical_constant = z3.Const(current_var_name, self.TypeSort)
        return z3.And(*(z3.Implies(type_rel(previous_var_logical_constant), type_rel(current_var_logical_constant))
                        for type_rel in self._possible_type_relations))
        
    def var_logical_name(self, var_name, program_location):
        return "%s_%d" % (var_name, self._cfg.get_instruction_number(program_location))
    
    def _type_preserved_for_variables(self, concrete_transformation, variables_names_unchanged):
        all_types_preserved_formula = True
        for var_name in variables_names_unchanged:
            # TODO: perhaps convert CFG to SSA and all of this becomes trivial?
            previous_var_name = self.var_logical_name(var_name, concrete_transformation.current_location())
            current_var_name  = self.var_logical_name(var_name, concrete_transformation.next_location()) 
            all_types_preserved_formula = z3.And(all_types_preserved_formula, 
                                                self._type_propagated_formula(previous_var_name, current_var_name))
        return all_types_preserved_formula
        
    def _no_change(self, concrete_transformation, abstract_type):
        return AbstractType(self._type_preserved_for_variables(concrete_transformation, 
                                                               self._cfg.program_vars()))
    
    def _assign(self, concrete_transformation, abstract_type):
        assigned_to_var_name = concrete_transformation.current_location().gens[0]
        assigned_from_var_name = concrete_transformation.current_location().uses[0]
        
        all_vars_except_changed = self._cfg.program_vars()
        all_vars_except_changed.remove(assigned_to_var_name)
        
        formula = z3.And(self._type_preserved_for_variables(concrete_transformation, all_vars_except_changed),
                         self._type_propagated_formula(assigned_from_var_name, assigned_to_var_name))
        
        return AbstractType(formula)

    def transform(self, concrete_trasformation, abstract_type):
        instruction = concrete_trasformation.current_location()
        op = instruction.opcode
        if op == tac.OP.NOP:
            return self._no_change(concrete_trasformation, abstract_type)
        elif op == tac.OP.ASSIGN:
            return self._assign(concrete_trasformation, abstract_type)
        elif op == tac.OP.IMPORT:
            return self._no_change(concrete_trasformation, abstract_type)
        elif op == tac.OP.BINARY:
            # TODO: do something real
            return self._assign(concrete_trasformation, abstract_type)
            #return self._binary_operator(concrete_trasformation, abstract_type)
        elif op == tac.OP.INPLACE:
            assert False
        elif op == tac.OP.CALL:
            assert False, "Function calls not supported"
        elif op == tac.OP.JUMP:
            return self._no_change(concrete_trasformation, abstract_type)
        elif op == tac.OP.FOR:
            assert False
        elif op == tac.OP.RET:
            assert False, "Function calls not supported"
        else:
            assert False, "Unknown Three Address Code instruction in %s" % str(instruction) 
            
    def equiv_type(self, abstract_type1, abstract_type2):
        # TODO: note that in the current implementation this has little use,
        # since in the chaotic iterations we really need to traverse each edge just once
        solver = z3.Solver()
        solver.add(z3.Not(z3.And(z3.Implies(abstract_type1.formula(), abstract_type2.formula()), 
                                         z3.Implies(abstract_type2.formula(), abstract_type1.formula()))))
        res = solver.check()
        assert(res == z3.sat or res == z3.unsat), res
        return res == z3.unsat
        
    def join(self, abstract_type1, abstract_type2):
        return AbstractType(z3.Or(abstract_type1.formula(), abstract_type2.formula()))