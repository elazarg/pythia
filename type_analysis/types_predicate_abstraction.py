import tac
import z3
import instruction_utils
import z3_utils

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
        self._numeric_rel = z3.Function('numeric', self.TypeSort, z3.BoolSort())
        self._string_rel = z3.Function('string', self.TypeSort, z3.BoolSort())
        self._possible_type_relations = [self._numeric_rel,
                                        self._string_rel]
            
    def lattice_bottom(self):
        return AbstractType(formula=False)
        

    def _type_logical_const(self, var_name):
        return z3.Const(var_name, self.TypeSort)

    def _type_propagated_formula(self, previou_var_name, current_var_name):
        previous_var_logical_constant = z3.Const(previou_var_name, self.TypeSort)
        current_var_logical_constant = self._type_logical_const(current_var_name)
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
        # TODO: must refactor: instructions to classes?
        assigned_to_var_name = concrete_transformation.current_location().gens[0]
        change_to_assigned = False
        expression_assigned = concrete_transformation.current_location().uses[0]
        
        assigned_var_logical_constant = self._type_logical_const(self.var_logical_name(assigned_to_var_name,
                                                                                       concrete_transformation.next_location()))
        
        if instruction_utils.is_var_reference(expression_assigned):
            assigned_from_var_name = expression_assigned
            # TODO: var_logical_name is important and buggy, make sure we don't make this mistake somehow
            assigned_from_name_with_context = self.var_logical_name(assigned_from_var_name, concrete_transformation.current_location())
            assigned_to_name_with_context = self.var_logical_name(assigned_to_var_name, concrete_transformation.next_location())
            change_to_assigned = self._type_propagated_formula(assigned_from_name_with_context, 
                                                               assigned_to_name_with_context)
        elif instruction_utils.is_numeric_literal(expression_assigned):
            change_to_assigned = self._numeric_rel(assigned_var_logical_constant)
        elif instruction_utils.is_string_literal(expression_assigned):
            change_to_assigned = self._string_rel(assigned_var_logical_constant)
        
        all_vars_except_changed = self._cfg.program_vars()
        all_vars_except_changed.remove(assigned_to_var_name)
        
        formula = z3.And(self._type_preserved_for_variables(concrete_transformation, all_vars_except_changed),
                         change_to_assigned)
        
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
        # Note: In the current implementation this has little use,
        # since in the chaotic iterations we really need to traverse each edge just once
        if z3_utils.sat_formula(z3.And(abstract_type1.formula(), z3.Not(abstract_type2.formula()))):
            return False
        if z3_utils.sat_formula(z3.And(abstract_type2.formula(), z3.Not(abstract_type1.formula()))):
            return False
        return True
        
    def join(self, abstract_type1, abstract_type2):
        return AbstractType(z3.Or(abstract_type1.formula(), abstract_type2.formula()))
    
    def _operand_is_numeric_formula(self, operand_expr, program_location):
        if instruction_utils.is_var_reference(operand_expr):
            operand_logical_const = self._type_logical_const(self.var_logical_name(operand_expr, program_location))
            return self._numeric_rel(operand_logical_const)
        elif instruction_utils.is_numeric_literal(operand_expr):
            return True
        elif instruction_utils.is_string_literal(operand_expr):
            return False
        
    def _operand_is_string_formula(self, operand_expr, program_location):
        if instruction_utils.is_var_reference(operand_expr):
            operand_logical_const = self._type_logical_const(self.var_logical_name(operand_expr, program_location))
            return self._string_rel(operand_logical_const)
        elif instruction_utils.is_numeric_literal(operand_expr):
            return False
        elif instruction_utils.is_string_literal(operand_expr):
            return True
    
    def _binary_op_constraint(self, program_location):
        op = program_location.op
        operands = program_location.uses
        
        all_operands_numeric_formula = z3.And(*(self._operand_is_numeric_formula(operand, program_location)
                                                for operand in operands))
        all_operands_string_formula = z3.And(*(self._operand_is_string_formula(operand, program_location)
                                                for operand in operands))
        if op in ['**', '*', '//', '/', '%', '-', '<<', '>>']:
            return all_operands_numeric_formula
        elif op == '+':
            return z3.Or(all_operands_numeric_formula, all_operands_string_formula)
        else:
            assert False, "Unknown type safety: op %s of instruction %s" % (op, program_location)
    
    def generate_safety_constraints(self, program_location):
        instruction = program_location
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
            assert False, "Function calls not supported %s" % str(instruction)
        elif opcode == tac.OP.JUMP:
            return True # TODO:
        elif opcode == tac.OP.FOR:
            assert False
        elif opcode == tac.OP.RET:
            if instruction.uses[0] == 'None':
                return True
            assert False, "Function calls not supported %s" % str(instruction)
        else:
            assert False, "Unknown Three Address Code instruction in %s" % str(instruction) 
            
    def get_types_exclusion_theory(self):
        return set()