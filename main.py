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
		n_envs, 
		iter_num, 
		seed, 
		net_arch_dim, 
		obstacle_num, 
		gnn_type, 
		early_stop):

	algo_name = algo
	log_name = algo_name

	if early_stop:
		target_kl = 0.005
	else:
		target_kl = None

	gnn_which = None
	if 'GNN' in algo:
		gnn_which = gnn_type
		log_name += gnn_type

	algo = eval('fam_gnn.'+algo)
	if 'Turtlebot' in env_id:
		# env_kwargs = {'obstacle_num': obstacle_num, 'use_gui': True}
		env_kwargs = {'obstacle_num': obstacle_num}
	else:
		env_kwargs = None

	env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
	# make experiment directory
	logdir = f"{env_id}+n_obstalces={obstacle_num}/{log_name}/logs/{int(time.time())}/"
	modeldir = f"{env_id}+n_obstalces={obstacle_num}/{log_name}/models/{int(time.time())}/"

	if not os.path.exists(modeldir):
		os.makedirs(modeldir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)
		
	model = algo(
				policy_type, 
	      		env, 
				verbose=1, 
				tensorboard_log=logdir, 
				net_arch_dim=net_arch_dim, 
				obstacle_num=obstacle_num, 
				gnn_type=gnn_which, 
				seed=seed,
				target_kl=target_kl)

	for i in range(iter_num):
		model.learn(reset_num_timesteps=False, tb_log_name=f"{algo_name}")
		model.save(modeldir, f'{i * n_envs * model.n_steps}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='Turtlebot-v2') # 'Turtlebot-v2''Safexp-PointGoal1-v0'
    parser.add_argument('--algo', type=str, default='GNN_PPO') 
    parser.add_argument('--policy_type', type=str, default='MlpPolicy')
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--iter_num', type=int, default=500) # Total_timestep = iter_num * n_envs * n_steps, here is 2000 * 4 * 20480 = 1.2e7
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--net_arch_dim', type=int, default=64)
    parser.add_argument('--obstacle_num', type=int, default=5)
    parser.add_argument('--gnn_type', type=str, default='fam_rel_gcn') 
    # fam_gnn, fam_gnn_noatte, gat, rel_gcn, fam_rel_gcn | temp_fam_gnn, temp_fam_rel_gcn
    parser.add_argument('--early_stop', action='store_true') # if no action, or said default if False, otherwise it's True
    args = parser.parse_args()
    
    main(
	    args.env_id, 
		args.algo, 
		args.policy_type, 
		args.n_envs, 
		args.iter_num, 
		args.seed,
		args.net_arch_dim,
		args.obstacle_num,
		args.gnn_type,
		args.early_stop)
