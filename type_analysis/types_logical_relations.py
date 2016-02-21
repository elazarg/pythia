import instruction_utils
import z3

TypeSort = z3.DeclareSort("Type")

class TypeRelation(object):
    # TODO: use SSA and eliminate dependence on cfg & program location
    def __init__(self, type_relation_symbol_name, cfg):
        super(TypeRelation, self).__init__()
        
        self._type_relation_symbol = z3.Function(type_relation_symbol_name, TypeSort, z3.BoolSort())
        
        self._cfg = cfg
        
    def var_logical_name(self, var_name, program_location):
        return "%s_%d" % (var_name, self._cfg.get_instruction_number(program_location))
    
    def _type_logical_const(self, var_name):
        return z3.Const(var_name, TypeSort)
        
    def has_type_of_formula(self, expression, program_location):
        assert instruction_utils.is_var_reference(expression)
        return self._type_relation_symbol(self._type_logical_const(self.var_logical_name(expression, program_location)))
    
class NumericRelation(TypeRelation):
    def __init__(self, cfg):
        super(NumericRelation, self).__init__('numeric', cfg)
        
    def has_type_of_formula(self, expression, program_location):
        if instruction_utils.is_numeric_literal(expression):
            return True
        if instruction_utils.is_var_reference(expression):
            return TypeRelation.has_type_of_formula(self, expression, program_location)
        return False
    
class StringRelation(TypeRelation):
    def __init__(self, cfg):
        super(StringRelation, self).__init__('string', cfg)
        
    def has_type_of_formula(self, expression, program_location):
        if instruction_utils.is_string_literal(expression):
            return True
        if instruction_utils.is_var_reference(expression):
            return TypeRelation.has_type_of_formula(self, expression, program_location)
        return False