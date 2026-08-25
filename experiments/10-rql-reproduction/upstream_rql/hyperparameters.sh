python main.py --agent=agents/rql.py --env_name=scene-play-singletask-v0 --sparse --agent.alpha=3.0 --agent.expectile=0.7 --agent.ensemble_ct=10 --agent.rho=0.5 --agent.h=5 --offline_steps=1000000 --online_steps=0 --agent.batch_size=256

python main.py --agent=agents/rql.py --env_name=puzzle-3x3-play-singletask-v0 --sparse --agent.alpha=1.0 --agent.expectile=0.7 --agent.ensemble_ct=10 --agent.rho=0.5 --agent.h=5 --offline_steps=1000000 --online_steps=0 --agent.batch_size=256

python main.py --agent=agents/rql.py --env_name=puzzle-4x4-play-singletask-v0 --sparse --ogbench_dataset_dir=<> --agent.alpha=1.0 --agent.expectile=0.9 --agent.ensemble_ct=10 --agent.rho=0.5 --agent.h=5 --offline_steps=1000000 --online_steps=0 --agent.batch_size=256 

python main.py --agent=agents/rql.py --env_name=cube-double-play-singletask-v0 --agent.alpha=10.0 --agent.expectile=0.9 --agent.ensemble_ct=10 --agent.rho=0.5 --agent.h=5 --offline_steps=1000000 --online_steps=0 --agent.batch_size=256 

python main.py --agent=agents/rql.py --env_name=cube-triple-play-singletask-v0 --agent.alpha=1.0 --agent.expectile=0.9 --agent.ensemble_ct=10 --agent.rho=0.5 --agent.h=5 --offline_steps=1000000 --online_steps=0 --agent.batch_size=256 

python main.py --agent=agents/rql.py --env_name=cube-quadruple-play-singletask-v0 --ogbench_dataset_dir=<> --agent.alpha=1.0 --agent.expectile=0.7 --agent.ensemble_ct=10 --agent.rho=0.5 --agent.h=5 --offline_steps=1000000 --online_steps=0 --agent.batch_size=256 

python main.py --agent=agents/rql.py --env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=0.1 --agent.expectile=0.5 --agent.ensemble_ct=10 --agent.rho=0.5 --agent.h=1 --offline_steps=1000000 --online_steps=0 --agent.batch_size=256 

python main.py --agent=agents/rql.py --env_name=antmaze-giant-navigate-singletask-v0 --agent.alpha=0.1 --agent.expectile=0.5 --agent.ensemble_ct=10 --agent.rho=0.5 --agent.h=1 --agent.discount=0.995 --offline_steps=1000000 --online_steps=0 --agent.batch_size=256

python main.py --agent=agents/rql.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.alpha=0.3 --agent.expectile=0.5 --agent.ensemble_ct=10 --agent.rho=0.0 --agent.h=1 --agent.discount=0.995 --offline_steps=1000000 --online_steps=0 --agent.batch_size=256

python main.py --agent=agents/rql.py --env_name=humanoidmaze-large-navigate-singletask-v0 --agent.alpha=0.3 --agent.expectile=0.5 --agent.ensemble_ct=10 --agent.rho=0.0 --agent.h=1 --agent.discount=0.995 --offline_steps=1000000 --online_steps=0 --agent.batch_size=256

