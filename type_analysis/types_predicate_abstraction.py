import tac
import z3
import z3_utils
from types_logical_relations import NumericRelation, StringRelation, BoolRelation

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
        
    def formula(self):
        return self._formula
    
    def join_with(self, abstract_type):
        self._formula = z3.Or(self._formula, abstract_type._formula)
        self._simplify()
        
    def _simplify(self):
        if self._formula in [True, False]:
            return
        self._formula = z3.simplify(self._formula)
        
    def __str__(self):
        return "AbstractType(%s)" % str(self._formula)

class AbstractTypesAnalysis(object):
    def __init__(self, cfg):
        super(AbstractTypesAnalysis, self).__init__()
        
        self._cfg = cfg
                
        self._numeric_rel = NumericRelation(self._cfg)
        self._string_rel = StringRelation(self._cfg)
        self._bool_rel = BoolRelation(self._cfg)
        
        self._possible_type_relations = [self._numeric_rel,
                                        self._string_rel,
                                        self._bool_rel]
            
    def lattice_bottom(self):
        return AbstractType(formula=False)
    
    def _operation_type_implies(self, concrete_transformation, ret_var, ret_type, operands_with_types):
        return z3.Implies(z3.And(*(type_relation.has_type_of_formula(operand_expr, concrete_transformation.current_location())
                                for (operand_expr, type_relation) in operands_with_types)),
                          ret_type.has_type_of_formula(ret_var, concrete_transformation.next_location()))

    def _type_propagated_formula(self, expression_propagated_from, current_var_name, concrete_transformation):
        return z3.And(*[self._operation_type_implies(concrete_transformation, current_var_name, type_rel,
                                                    [(expression_propagated_from, type_rel)])
                        for type_rel in self._possible_type_relations])

    def _type_preserved_for_variables(self, concrete_transformation, variables_names_unchanged):
        all_types_preserved_formula = True
        for var_name in variables_names_unchanged: 
            all_types_preserved_formula = z3.And(all_types_preserved_formula, 
                                                self._type_propagated_formula(var_name, var_name, concrete_transformation))
        return all_types_preserved_formula
        
    def _no_change(self, concrete_transformation, abstract_type):
        return AbstractType(self._type_preserved_for_variables(concrete_transformation, 
                                                               self._cfg.program_vars()))
    
    def _assign(self, concrete_transformation, abstract_type):
        assigned_to_var_name = concrete_transformation.current_location().instruction().gens[0]
        expression_assigned = concrete_transformation.current_location().instruction().uses[0]
        
        change_to_assigned = self._type_propagated_formula(expression_assigned, assigned_to_var_name, concrete_transformation)
        
        all_vars_except_changed = self._cfg.program_vars()
        all_vars_except_changed.remove(assigned_to_var_name)
        
        formula = z3.And(self._type_preserved_for_variables(concrete_transformation, all_vars_except_changed),
                         change_to_assigned)
        return AbstractType(formula)
    
    def _function_call_typing(self, concrete_transformation, ret_var, function_name, operands):
        all_string_implies_string = self._operation_type_implies(concrete_transformation, 
            ret_var, self._string_rel, 
            [(operand, self._string_rel) for operand in operands])
        
        all_numeric_implies_numeric = self._operation_type_implies(concrete_transformation, 
            ret_var, self._numeric_rel, 
            [(operand, self._numeric_rel) for operand in operands])
        
        all_boolean_implies_boolean = self._operation_type_implies(concrete_transformation, 
            ret_var, self._bool_rel, 
            [(operand, self._bool_rel) for operand in operands])
        
        if function_name in ['**', '*', '//', '/', '%', '-', '<<', '>>']:
            return all_numeric_implies_numeric
        if function_name == '+':
            return z3.Or(all_numeric_implies_numeric, all_string_implies_string)
        if function_name in ['&', '|', '^']:
            # TODO: not support implicit conversion to boolean
            return all_boolean_implies_boolean
        
        if function_name == 'not':
            return all_boolean_implies_boolean
        else:
            assert False, "Unknown function call"
            
    def _binary(self, concrete_transformation, abstract_type):
        instruction = concrete_transformation.current_location().instruction()
        op = instruction.op
        (assigned_to_var,) = instruction.gens
        (operand1, operand2) = instruction.uses
        return AbstractType(self._function_call_typing(concrete_transformation, assigned_to_var, op, [operand1, operand2]))
    
    def _call(self, concrete_transformation, abstract_type):
        instruction = concrete_transformation.current_location().instruction()
        args = instruction.uses[1:]  # TODO: add kargs (see TAC)
        ret_var = instruction.gens[0]
        function_name = instruction.func
        return AbstractType(self._function_call_typing(concrete_transformation, ret_var, function_name, args))

    def transform(self, concrete_transformation, abstract_type):
        instruction = concrete_transformation.current_location().instruction()
        op = instruction.opcode
        if op == tac.OP.NOP:
            return self._no_change(concrete_transformation, abstract_type)
        elif op == tac.OP.ASSIGN:
            return self._assign(concrete_transformation, abstract_type)
        elif op == tac.OP.IMPORT:
            return self._no_change(concrete_transformation, abstract_type)
        elif op == tac.OP.BINARY:
            return self._binary(concrete_transformation, abstract_type)
        elif op == tac.OP.INPLACE:
            assert False
        elif op == tac.OP.CALL:
            return self._call(concrete_transformation, abstract_type)
        elif op == tac.OP.JUMP:
            return self._no_change(concrete_transformation, abstract_type)
        elif op == tac.OP.FOR:
            assert False
        elif op == tac.OP.RET:
            assert False, "Function calls not supported"
        elif op == tac.OP.DEL:
            return self._no_change(concrete_transformation, abstract_type)
        else:
            assert False, "Unknown Three Address Code instruction in %s" % str(instruction) 
            
    def equiv_type(self, abstract_type1, abstract_type2):
        # Note: In the current implementation this has little use,
        # since in the chaotic iterations we really need to traverse each edge just once
        if z3_utils.sat_formula(z3.And(abstract_type1.formula(), z3.Not(abstract_type2.formula()))):
            return False
        if z3_utils.sat_formula(z3.And(abstract_type2.formula(), z3.Not(abstract_type1.formula()))):
            return False
        return True
        
    def join(self, abstract_type1, abstract_type2):
        return AbstractType(z3.Or(abstract_type1.formula(), abstract_type2.formula()))
    
    def _binary_op_constraint(self, program_location):
        op = program_location.instruction().op
        operands = program_location.instruction().uses
        
        all_operands_numeric_formula = z3.And(*(self._numeric_rel.has_type_of_formula(operand, program_location)
                                                for operand in operands))
        all_operands_string_formula = z3.And(*(self._string_rel.has_type_of_formula(operand, program_location)
                                                for operand in operands))
        all_operands_bool_formula = z3.And(*(self._bool_rel.has_type_of_formula(operand, program_location)
                                                for operand in operands))
        if op in ['**', '*', '//', '/', '%', '-', '<<', '>>']:
            return all_operands_numeric_formula
        elif op == '+':
            return z3.Or(all_operands_numeric_formula, all_operands_string_formula)
        elif op in ['&', '|', '^']: 
            return all_operands_bool_formula
        else:
            assert False, "Unknown type safety: op %s of instruction %s" % (op, program_location)
            
    def _call_constraints(self, program_location):
        instruction = program_location.instruction()
        function_name = instruction.func
        args = instruction.uses[1:]  # TODO: add kargs (see TAC)
        
        all_args_bool_formula = z3.And(*(self._bool_rel.has_type_of_formula(operand, program_location)
                                                for operand in args))
        
        if function_name == 'not':
            print(all_args_bool_formula)
            return all_args_bool_formula
        assert False, "Unknown function call for type constraints"
    
    def generate_safety_constraints(self, program_location):
        instruction = program_location.instruction()
        opcode = instruction.opcode
        if opcode == tac.OP.NOP:
            return True
        elif opcode == tac.OP.ASSIGN:
            return True
        elif opcode == tac.OP.IMPORT:
            return True
        elif opcode == tac.OP.BINARY:
            return self._binary_op_constraint(program_location)
        elif opcode == tac.OP.INPLACE:
            assert False
        elif opcode == tac.OP.CALL:
            return self._call_constraints(program_location)
        elif opcode == tac.OP.JUMP:
            # TODO: not supporting implicit conversion to bool
            return self._bool_rel.has_type_of_formula(program_location.instruction().uses[0], program_location)
        elif opcode == tac.OP.FOR:
            assert False
        elif opcode == tac.OP.RET:
            if instruction.uses[0] == 'None':
                return True
            assert False, "Function calls not supported %s" % str(instruction)
        elif opcode == tac.OP.DEL:
            return True
        else:
            assert False, "Unknown Three Address Code instruction in %s" % str(instruction) 
            
    def get_types_exclusion_theory(self):
        types_exclusion_theory = set()
        for program_location in self._cfg.all_locations():
            for var_name in self._cfg.program_vars():
                # TODO: should be more fine-grained when we support subtyping
                for possible_type1, possible_type2 in zip(self._possible_type_relations, self._possible_type_relations):
                    if possible_type1 != possible_type2:
                        exclusion_statement = z3.Not(z3.And(possible_type1.has_type_of_formula(var_name, program_location),
                                                            possible_type1.has_type_of_formula(var_name, program_location)))
                        types_exclusion_theory.add(exclusion_statement)
        return types_exclusion_theory