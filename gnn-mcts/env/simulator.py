import time
from env.tsp_env import TSPEnv


class Simulator:
    def __init__(self, env: TSPEnv):
        self.env = env

    def start(self, player, rank, episode):
        state = self.env.initial_state()
        total_s_t = time.time()
        def print_tree_node(rt):
            print('tour:', rt["state"]["tour"], '#children:', len(rt["_children"].keys()), 'n_visits:', rt["_n_visits"], 'Q:', rt["_Q"], 'u:', rt["_u"], 'P:', rt["_P"], 'n_vlosses:', rt["n_vlosses"])
        def print_all_children(rt, level=0):
            print("Level:", level)
            print_tree_node(rt)
            child_list = list(rt["_children"].keys())
            for i in range(len(child_list)):
              rt_child = print_all_children(rt["_children"][child_list[i]].__dict__, level+1)

        print("**************Before loop:")
        rt = player.mcts._root.__dict__
        print_all_children(rt)

        while True:
            s_t = time.time()
            move, best_sol = player.get_action()
            state = self.env.next_state(state, move)
            print(
                "Process %2d, episode %d, time %2f ---> %3d, best %4f" % (
                    rank, episode, time.time() - s_t, len(state['tour']), best_sol))
            rt = player.mcts._root.__dict__
            print_all_children(rt)
            game_end = self.env.is_done_state(state)
            if game_end:
                print(time.time() - total_s_t)
                return state['tour'], self.env.get_return(state)
