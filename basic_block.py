from itertools import accumulate
from typing import List  # :)  

import instruction as ins
from instruction import BCode
import operator as op
import utils 


class BasicBlock(list):
    'Try to keep this class about itself, not about the state of the graph'
    def __init__(self, block):
        super().__init__(block)
        self.stack_effect_acc = list(accumulate(b.stack_effect() for b in self))
        self.stack_effect = self.stack_effect_acc[-1]
        
        "it's there so we can make sure that the stack size is never negative"
        self.minimum_in_stack_depth = max(0, -min(self.stack_effect_acc))
    
    @property
    def offset(self):
        return self[0].offset
    
    @property
    def last(self):
        return self[-1]
        
    def next_list(self):
        return [(target, self.stack_effect)
                for target, _se in self.last.next_list()]

    def absolute_operand_depth(self, depth_in):
        return [depth_in] + [depth_in + d for d in self.stack_effect_acc]


class ForIterBasicBlock(BasicBlock):
    ''' FOR_ITER is both target and source of a jump.
    FOR_ITER(delta):
        TOS is an iterator. Call its __next__() method.
        If this yields a new value, push it on the stack (leaving the iterator below it).
        If the iterator indicates it is exhausted TOS is popped, and the byte code counter is incremented by delta.       
    '''
    def next_list(self):
        # we can do it only because we have a single instruction
        return self.last.next_list()


def makeBasicBlock(block):
    if block[0].is_for_iter:
        return ForIterBasicBlock(block)
    return BasicBlock(block)


def split_to_basic_blocks(bs: List[BCode]) -> List[BasicBlock]:
    bss = utils.partition(bs, before=op.attrgetter('is_jump_target'),
                              after=op.attrgetter('is_jump_source'))
    return list(makeBasicBlock(b) for b in bss)


def prepare(f: 'e.g. function'):
    return split_to_basic_blocks(ins.get_instructions(f))


def print_blocks(blocks: List[BasicBlock]):
    for block in blocks:
        print(block.offset, ':',
              block.next_list(), end=' <- ')
        print(*['({0.opname}, {0.argval}, {0.offset})'.format(x)
                for x in block])

def test():
    blocks = prepare(utils.partition)
    print_blocks(blocks)

if __name__ == '__main__':
    test()
