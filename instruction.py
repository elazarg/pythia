import dis, inspect

class BCode(dis.Instruction):    
    @property
    def is_sequencer(self):
        return 'RETURN' in self.opname \
            or 'CONTINUE' in self.opname \
            or 'BREAK' in self.opname \
            or self.opname in ['JUMP_FORWARD', 'JUMP_ABSOLUTE', 'END_FINALLY']
            
    @property
    def is_jump_source(self):
        return self.is_jump or self.is_sequencer or self is OP_START
        # yield does not interrupt the flow
        # exceptions do not count, since they can occur practically anywhere
            
    @property
    def is_jump(self):
        return self.opcode in dis.hasjrel + dis.hasjabs \
               # or  'JUMP' in self.opname
               
    @property
    def size(self):
        return 1 if self.opcode < dis.HAVE_ARGUMENT else 3
        
    def next_list(self):
        next_offset = self.offset + self.size
        if self.opname == 'FOR_ITER':
            return [(next_offset, self.stack_effect()),
                    (self.argval, -1)]
        res = []
        if not self.is_sequencer:
            res.append( (next_offset, self.stack_effect() ) )
        if self.is_jump_source and self is not OP_START:
            res.append( (self.argval, self.stack_effect()) )
        return res
    
    @property 
    def is_block_boundary(self):
        return self.is_jump_source or self.is_jump_target
             
    def stack_effect(self):
        if self is OP_START: return 0
        return dis.stack_effect(self.opcode, self.arg) 
     
    @property
    def is_for_iter(self):
        return self.opname == 'FOR_ITER'
    
def source(f):
    return inspect.getsource(f)

def get_instructions(f):
    res =[OP_START] + [BCode(*i) for i in dis.get_instructions(f)]
    # res.append(make_end(next(res[-1].next_list)))
    return res

OP_START = BCode(opname='START', opcode=-1, arg=None, argval=0, argrepr=None, offset=-1, starts_line=-1, is_jump_target=False)

def make_end(offset):
    return BCode(opname='END', opcode=-2, arg=None, argval=None, argrepr=None, offset=offset, starts_line=None, is_jump_target=False)



def example():
    x = 1
    for x in range(7):
        print(x)

if __name__ == '__main__':   
    for b in get_instructions(example):
        print(b)

''' 
<0>
POP_TOP
ROT_TWO
ROT_THREE
DUP_TOP
DUP_TOP_TWO
<6>
<7>
<8>
NOP
UNARY_POSITIVE
UNARY_NEGATIVE
UNARY_NOT
<13>
<14>
UNARY_INVERT
BINARY_MATRIX_MULTIPLY
INPLACE_MATRIX_MULTIPLY
<18>
BINARY_POWER
BINARY_MULTIPLY
<21>
BINARY_MODULO
BINARY_ADD
BINARY_SUBTRACT
BINARY_SUBSCR
BINARY_FLOOR_DIVIDE
BINARY_TRUE_DIVIDE
INPLACE_FLOOR_DIVIDE
INPLACE_TRUE_DIVIDE
<30>
<31>
<32>
<33>
<34>
<35>
<36>
<37>
<38>
<39>
<40>
<41>
<42>
<43>
<44>
<45>
<46>
<47>
<48>
<49>
GET_AITER
GET_ANEXT
BEFORE_ASYNC_WITH
<53>
<54>
INPLACE_ADD
INPLACE_SUBTRACT
INPLACE_MULTIPLY
<58>
INPLACE_MODULO
STORE_SUBSCR
DELETE_SUBSCR
BINARY_LSHIFT
BINARY_RSHIFT
BINARY_AND
BINARY_XOR
BINARY_OR
INPLACE_POWER
GET_ITER
GET_YIELD_FROM_ITER
PRINT_EXPR
LOAD_BUILD_CLASS
YIELD_FROM
GET_AWAITABLE
<74>
INPLACE_LSHIFT
INPLACE_RSHIFT
INPLACE_AND
INPLACE_XOR
INPLACE_OR
BREAK_LOOP
WITH_CLEANUP_START
WITH_CLEANUP_FINISH
RETURN_VALUE
IMPORT_STAR
<85>
YIELD_VALUE
POP_BLOCK
END_FINALLY
POP_EXCEPT
STORE_NAME
DELETE_NAME
UNPACK_SEQUENCE
FOR_ITER
UNPACK_EX
STORE_ATTR
DELETE_ATTR
STORE_GLOBAL
DELETE_GLOBAL
<99>
LOAD_CONST
LOAD_NAME
BUILD_TUPLE
BUILD_LIST
BUILD_SET
BUILD_MAP
LOAD_ATTR
COMPARE_OP
IMPORT_NAME
IMPORT_FROM
JUMP_FORWARD
JUMP_IF_FALSE_OR_POP
JUMP_IF_TRUE_OR_POP
JUMP_ABSOLUTE
POP_JUMP_IF_FALSE
POP_JUMP_IF_TRUE
LOAD_GLOBAL
<117>
<118>
CONTINUE_LOOP
SETUP_LOOP
SETUP_EXCEPT
SETUP_FINALLY
<123>
LOAD_FAST
STORE_FAST
DELETE_FAST
<127>
<128>
<129>
RAISE_VARARGS
CALL_FUNCTION
MAKE_FUNCTION
BUILD_SLICE
MAKE_CLOSURE
LOAD_CLOSURE
LOAD_DEREF
STORE_DEREF
DELETE_DEREF
<139>
CALL_FUNCTION_VAR
CALL_FUNCTION_KW
CALL_FUNCTION_VAR_KW
SETUP_WITH
EXTENDED_ARG
LIST_APPEND
SET_ADD
MAP_ADD
LOAD_CLASSDEREF
BUILD_LIST_UNPACK
BUILD_MAP_UNPACK
BUILD_MAP_UNPACK_WITH_CALL
BUILD_TUPLE_UNPACK
BUILD_SET_UNPACK
SETUP_ASYNC_WITH
<155>
<156>
<157>
<158>
<159>
<160>
<161>
<162>
<163>
<164>
<165>
<166>
<167>
<168>
<169>
<170>
<171>
<172>
<173>
<174>
<175>
<176>
<177>
<178>
<179>
<180>
<181>
<182>
<183>
<184>
<185>
<186>
<187>
<188>
<189>
<190>
<191>
<192>
<193>
<194>
<195>
<196>
<197>
<198>
<199>
<200>
<201>
<202>
<203>
<204>
<205>
<206>
<207>
<208>
<209>
<210>
<211>
<212>
<213>
<214>
<215>
<216>
<217>
<218>
<219>
<220>
<221>
<222>
<223>
<224>
<225>
<226>
<227>
<228>
<229>
<230>
<231>
<232>
<233>
<234>
<235>
<236>
<237>
<238>
<239>
<240>
<241>
<242>
<243>
<244>
<245>
<246>
<247>
<248>
<249>
<250>
<251>
<252>
<253>
<254>
<255>
'''