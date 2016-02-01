#!/sbin/python3
import ast

class Visitor(ast.NodeVisitor):
    def __init__(self, parent = None):
        self.parent = parent
    
    def lookup(self, name):
        return self.sym.get_var(name)
    
    def bind_weak(self, var_id, typeset):
        return self.sym.bind(var_id, typeset)
    
    def print(self):
        self.sym.print()
    
    def visit_AugAssign(self, ass):
        #potential bind occurrence
        pass
        
    def visit_Assign(self, ass):
        #potential bind occurrence
        pass
        
    def visit_Delete(self, delete):
        #potential bind occurrence
        pass
    
    def visit_Attribute(self, attr):
        return self.get_attr_types(attr.value, attr.attr)

    def visit_FunctionDef(self, func):
        pass
    
    def visit_ClassDef(self, cls):
        pass
            
    def visit_Call(self, value):
        pass
            
    def visit_IfExp(self, ifexp):
        r1 = self.visit(ifexp.body)
        r2 = self.visit(ifexp.orelse)
        res = r1.union(r2)
        return res
    
    def visit_Subscript(self, sub):
        assert False

    def visit_BinOp(self, binop):
        assert False
    
    def visit_If(self, stat):
        return self.visit_all_childs(stat)
    
    def visit_While(self, stat):
        pass
    
    def visit_For(self, stat):
        #bind occurrence
        pass
    
    def visit_Return(self, ret):
        pass

    def visit_Module(self, node):
        return self.run(node)
      
    def visit_Name(self, value):
        pass
    
    def visit_NameConstant(self, cons):
        pass

    def visit_ListComp(self, value):
        pass
    
    def visit_Lambda(self, lmb):
        returns = self.visit(lmb.body) 
        return self.create_func(lmb.args, returns, 'lambda')

    def visit_Expr(self, expr):
        self.visit(expr.value)
    
    def visit_arguments(self, args):
        pass

if __name__=='__main__':
    pass