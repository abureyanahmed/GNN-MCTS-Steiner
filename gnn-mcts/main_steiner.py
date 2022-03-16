import os
import time
import torch
import torch.multiprocessing as mp

from args import parser
from agent import Agent
from utils_steiner import load_data
from model.policy import TSPNetwork

os.environ["OMP_NUM_THREADS"] = "1"

def objective():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

    #test_graphs = load_data("data/{}/tsp{}.txt".format(args.graph_type, args.graph_size), args.graph_size)
    #test_graphs = load_data("data/random/tsp20/tsp20_train.txt", 20)
    #test_graphs = load_data("../segnn/dataset/random/tsp_20/processed/data_18994.pt", 20)
    #test_graphs = load_data("data/cluster/tsp100/tsp100_c_train.txt", 20)
    #test_graphs = load_data("data/cluster/tsp100/tsp100_c_train.txt", 5)
    test_graphs = load_data("../segnn/data/random/st15/st15_test.txt", 15)
    network = TSPNetwork(node_dim=args.node_dim, edge_dim=args.edge_dim, embed_dim=args.embed_dim,
                         hidden_dim=args.hidden_dim, graph_size=args.graph_size, layer=args.n_layer)

    if args.load:
        saved_state = torch.load(
            #"{}/{}/tsp{}.pth".format(args.load_model_dir, args.graph_type, args.graph_size),
            #"../segnn/saved/models/SE-GNN/1210_140144/checkpoint-epoch31.pth",
            #"../segnn/saved/models/SE-GNN/0201_194204/checkpoint-epoch41.pth",
            "../segnn/saved/models/SE-GNN/0303_121439/checkpoint-epoch60.pth",
            map_location=lambda storage, loc: storage)
        network.load_paras(saved_state)
        print("Load model successfully ~~")

    processes = []

    tasks_num = len(test_graphs) // args.n_worker
    extra_num = len(test_graphs) % args.n_worker

    # s_t = time.time()
    for idx in range(args.n_worker):
        if idx == args.n_worker - 1:
            graphs = test_graphs[idx * tasks_num: (idx + 1) * tasks_num + extra_num]
        else:
            graphs = test_graphs[idx * tasks_num: (idx + 1) * tasks_num]

        agent = Agent(idx, args, graphs, network)
        p = mp.Process(target=agent.run)
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()
    # print(time.time() - s_t)


if __name__ == '__main__':
    objective()
