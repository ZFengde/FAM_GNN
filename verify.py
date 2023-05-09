import gym
import turtlebot_env
import fam_gnn

import os
import time
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def main(
		env_id, 
		algo, 
		policy_type, 
		seed, 
		obstacle_num, 
		gnn_type):

	algo_name = algo
	seed = obstacle_num
	log_name = algo_name
	gnn_which = None
	
	if 'GNN' in algo:
		gnn_which = gnn_type
		log_name += gnn_type
		
	algo = eval('fam_gnn.'+algo)
	env = gym.make(env_id, use_gui=False, obstacle_num=obstacle_num)
	
	logdir = f"Test_log/{log_name+str(obstacle_num)}/logs/{int(time.time())}"
	modelpath = f"./model/{log_name}"
	if not os.path.exists(logdir):
		os.makedirs(logdir)
	model = algo(
				policy_type, 
	      		env, 
				verbose=1, 
				tensorboard_log=logdir, 
				net_arch_dim=64, 
				obstacle_num=obstacle_num, 
				gnn_type=gnn_which, 
				seed=seed)
	
	model.load(modelpath)
	model.test(env, 1000)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='Turtlebot-v3') # 'Turtlebot-v2''Safexp-PointGoal1-v0'
    parser.add_argument('--algo', type=str, default='GNN_PPO') 
    parser.add_argument('--policy_type', type=str, default='MlpPolicy')
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--obstacle_num', type=int, default=7)
    parser.add_argument('--indicator', type=int, default=2)
    parser.add_argument('--gnn_type', type=str, default='fam_rel_gcn') 
    # fam_gnn, fam_gnn_noatte, gat, rel_gcn, fam_rel_gcn | temp_fam_gnn, temp_fam_rel_gcn
    args = parser.parse_args()

    main(
	    args.env_id, 
		args.algo, 
		args.policy_type, 
		args.seed,
		args.obstacle_num,
		args.gnn_type)
