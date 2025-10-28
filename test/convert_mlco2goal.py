import imp
import numpy as np

if __name__ == "__main__":
    file_path = "atsp_uniform_100ins.txt"
    with open(file_path, "r") as f:
        lines = f.readlines()

    dists, sols = [], []
    for line in lines:
        dist, sol = line.strip().split("output")
        sol = sol.replace("\n", "")
        sol = np.array([int(x) for x in sol.split(" ")[1:]])
        num_nodes = sol.shape[0]-1 # exclude the last node
        dist = np.array([float(x) for x in dist.split("  ")[:-1]]).reshape(num_nodes, num_nodes)
        # 在dist 矩阵中多加一行，这一行是0行的copy,0列的copy也是0行的copy
        dist = np.hstack((dist, dist[0, :].reshape(-1,1)))
        dist = np.vstack((dist, dist[0, :]))
        
        dists.append(dist)
        sols.append(sol)
    
    dists = np.array(dists)
    sols = np.array(sols)
    opt_lens = []
    for i in range(dists.shape[0]):
        opt_len = 0
        for j in range(sols.shape[1] - 1):
            opt_len += dists[i, sols[i, j]-1, sols[i, j + 1]-1] # turn node index to 0-based
        opt_lens.append(opt_len)
    
    opt_lens = np.array(opt_lens)

    #save with npz
    np.savez("atsp_uniform_100ins.npz", dist_matrices=dists, sols=sols, tour_lens=opt_lens,is_training_dataset=False)