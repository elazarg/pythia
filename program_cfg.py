import networkx

def pairs(lst):
    return zip(lst, lst[1:])

class ProgramCFG(object):
    @staticmethod
    def tac_blocks_cfg_get_commands_in_block(tac_blocks_cfg, block_node, name):
        return tac_blocks_cfg.node[block_node][name]

    def __init__(self, tac_blocks_cfg, name):
        self._cfg = networkx.DiGraph()
        self._instruction_numbers = []
        
        for block_node in tac_blocks_cfg.nodes():
            commands_in_block_lst = ProgramCFG.tac_blocks_cfg_get_commands_in_block(tac_blocks_cfg, block_node, name)
            self.__initialize_nodes_and_intra_block_edges(commands_in_block_lst)
            
        self.__add_consecutive_blocks_edges(tac_blocks_cfg, name)
            
        self.__add_goto_edges(tac_blocks_cfg, name)
        
        first_block_id = sorted(tac_blocks_cfg.nodes())[0]
        self._start_location = tac_blocks_cfg.node[first_block_id][name][0]
        print(self._start_location)
    
    def __initialize_nodes_and_intra_block_edges(self, commands_in_block_lst):
            for tac_command in commands_in_block_lst:
                self._cfg.add_node(tac_command)
                # we rely here on the fact that two TAC objects, although they depict the same command,
                # do not equal to each other as they don't represent the same program location
                # TODO (Yotam): make this more explicit and less confusing in the instruction class
                # TODO: should be "program location"
                assert tac_command not in self._instruction_numbers
                self._instruction_numbers.append(tac_command)
            
            for tac_command, next_command in pairs(commands_in_block_lst):
                self._cfg.add_edge(tac_command, next_command)
                
    def __add_consecutive_blocks_edges(self, tac_blocks_cfg, name):
        for block_node_id in sorted(tac_blocks_cfg.nodes()):
            for succ_id in tac_blocks_cfg.successors(block_node_id):
                last_command_in_pre_block = ProgramCFG.tac_blocks_cfg_get_commands_in_block(tac_blocks_cfg, block_node_id, name)[-1]
                first_command_in_suc_block =  ProgramCFG.tac_blocks_cfg_get_commands_in_block(tac_blocks_cfg, succ_id, name)[0]
                self._cfg.add_edge(last_command_in_pre_block, first_command_in_suc_block)
                
    def __add_goto_edges(self, tac_blocks_cfg, name):
        for command in self._cfg.nodes():
            # TODO: handle also function calls
            target_block = command.target
            if target_block != None:
                target_block_node = tac_blocks_cfg.node[target_block]
                commands_in_target_block = target_block_node[name]
                assert commands_in_target_block != []
                # TODO: add identifier indicating "TRUE" part of the branch?
                self._cfg.add_edge(command, commands_in_target_block[0])
                
    def depict(self):
        for ins in networkx.dfs_preorder_nodes(self._cfg, self._start_location):
            cmd = ins.fmt.format(**ins._asdict())
            print(cmd , '\t\t', '' and ins)
            print("succ: ", [ins.format() for ins in self.successors(ins)])
            
    def get_instruction_number(self, instruction):
        return self._instruction_numbers.index(instruction)
    
    def initial_location(self):
        return self._start_location
    
    def program_vars(self):
        all_vars = set()
        for ins in self._cfg.nodes():
            all_vars.update(set(ins.gens))
            all_vars.update(set(ins.kills))

        return all_vars
    
    def all_locations(self):
        return self._cfg.nodes()
    
    def successors(self, program_location):
        return self._cfg.successors(program_location)