import networkx

def pairs(lst):
    return zip(lst, lst[1:])

class ProgramLocation(object):
    def __init__(self, instruction):
        self._instruction = instruction
        
    def instruction(self):
        return self._instruction
    
    def __str__(self):
        return "ProgramLocation(%s)" % self._instruction.format()

class ProgramCFG(object):
    @staticmethod
    def tac_blocks_cfg_get_commands_in_block(tac_blocks_cfg, block_node, name):
        return tac_blocks_cfg.node[block_node][name]

    def __init__(self, tac_blocks_cfg, name):
        self._cfg = networkx.DiGraph()
        self._instruction_numbers = []
        
        block_number_to_first_location = {}
        block_number_to_last_location = {}
        
        for block_node in tac_blocks_cfg.nodes():
            commands_in_block_lst = ProgramCFG.tac_blocks_cfg_get_commands_in_block(tac_blocks_cfg, block_node, name)
            program_locations_in_block = [ProgramLocation(instruction) for instruction in commands_in_block_lst]
            for program_location in program_locations_in_block:
                self._cfg.add_node(program_location)
                self._instruction_numbers.append(program_location)
            
            for program_location, next_location in pairs(program_locations_in_block):
                self._cfg.add_edge(program_location, next_location)
            
            block_number_to_first_location[block_node] = program_locations_in_block[0]
            block_number_to_last_location[block_node] = program_locations_in_block[-1]
            
        self.__add_consecutive_blocks_edges(tac_blocks_cfg, name,
                                            block_number_to_first_location, block_number_to_last_location)
        
        first_block_id = sorted(tac_blocks_cfg.nodes())[0]
        self._start_location = block_number_to_first_location[first_block_id]
        print(self._start_location)        
                
    def __add_consecutive_blocks_edges(self, tac_blocks_cfg, name,
                                       block_number_to_first_location, block_number_to_last_location):
        for block_node_id in sorted(tac_blocks_cfg.nodes()):
            for succ_id in tac_blocks_cfg.successors(block_node_id):
                last_command_in_pre_block = block_number_to_last_location.get(block_node_id)
                first_command_in_suc_block =  block_number_to_first_location.get(succ_id)
                self._cfg.add_edge(last_command_in_pre_block, first_command_in_suc_block)

    def depict(self):
        for location in networkx.dfs_preorder_nodes(self._cfg, self._start_location):
            cmd = location.instruction().fmt.format(**location.instruction()._asdict())
            print(cmd)
            print("succ: ", [suc_location.instruction().format() for suc_location in self.successors(location)])
            
    def get_instruction_number(self, instruction):
        return self._instruction_numbers.index(instruction)
    
    def initial_location(self):
        return self._start_location
    
    def program_vars(self):
        all_vars = set()
        for location in self._cfg.nodes():
            ins = location.instruction()
            # TODO: ignore constant literals
            all_vars.update(set(ins.gens))
            all_vars.update(set(ins.kills))

        return all_vars
    
    def all_locations(self):
        return self._cfg.nodes()
    
    def successors(self, program_location):
        return self._cfg.successors(program_location)