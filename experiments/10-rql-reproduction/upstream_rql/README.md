<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Reversal Q-Learning (RQL)</h1>
      <div style="height: 2px;"></div>
      <h2><a href="https://arxiv.org/abs/2606.17551">Paper</a> &emsp;</h2>
      <h2><a href="https://aober.ai/rql">Website</a> &emsp;</h2>
    </summary>
  </ul>
</div>

</div>

![teaser](assets/teaser.gif)



## Installation

RQL requires Python 3.9+ and is based on JAX. The main dependencies are
`jax >= 0.4.26`, `ogbench == 1.2.0`, and `gymnasium == 0.29.1`.
To install the full dependencies, simply run:
```bash
pip install -r requirements.txt
```


## Usage

The main implementation of RQL is in [agents/rql.py](agents/rql.py).

Tuned hyperparameters for each environment and agent are provided in the paper.
Complete list of RQL commands here: [hyperparameters.sh](hyperparameters.sh)

```bash

# RQL

python main.py 
    --agent=agents/rql.py 
    --env_name=humanoidmaze-large-navigate-singletask-v0 
    --agent.alpha=10 
    --agent.expectile=0.9 
    --agent.ensemble_ct=10 
    --agent.rho=0.0 
    --agent.h=1 
    --agent.discount=0.995 
    --offline_steps=1000000 
    --online_steps=0 
    --agent.batch_size=256
```



## Using larger datasets

The paper uses 100m-sized datasets for the OGBench puzzle-4x4 & cube-quadruple environments.
These datasets can be downloaded with the following commands (see [this section of the OGBench repository](https://github.com/seohongpark/ogbench?tab=readme-ov-file#additional-features) for more diverse 100M-sized datasets available):
```bash
# cube-quadruple-play-100m (100 datasets * 1000 length-1000 trajectories).
wget -r -np -nH --cut-dirs=2 -A "*.npz" https://rail.eecs.berkeley.edu/datasets/ogbench/cube-quadruple-play-100m-v0/
# puzzle-4x4-play-100m (100 datasets * 1000 length-1000 trajectories).
wget -r -np -nH --cut-dirs=2 -A "*.npz" https://rail.eecs.berkeley.edu/datasets/ogbench/puzzle-4x4-play-100m-v0/
```

</details>


## Acknowledgments

This codebase is built on top of reference implementations from [Flow Q-Learning](https://github.com/seohongpark/fql).


## Local formal-run entry point

This vendor copy is pinned to commit
`229c956efb4494c2b9bb0bbddbd67b761c93f1cc` and has a cluster-lifecycle patch
documented in [UPSTREAM_PROVENANCE.md](UPSTREAM_PROVENANCE.md). The formal
campaign invokes `main.py` with the task's unchanged command from
`hyperparameters.sh`, plus stable lifecycle fields:

```bash
python main.py \
  --run_dir=/absolute/path/to/runs/task_seed0 \
  --run_name=task_seed0 \
  --wandb_id=task_seed0 \
  --protocol_sha256=<canonical-campaign-sha256> \
  --gradient_checkpointing=true \
  --offline_steps=1000000 \
  --online_steps=0 \
  --env_name=<official-singletask-environment> \
  --agent=<this-directory>/agents/rql.py \
  <official per-task agent overrides>
```

The process exits 75 after a durable preemption/wall-time checkpoint and
resumes from `checkpoint.pkl` at the next absolute update. A successful run
writes `COMPLETED.json`; its `final_evaluation` object and `final_eval.csv`
contain the mandatory 50-episode result at the exact terminal update. The
current formal protocol uses exactly 1,000,000 offline updates. Array width is
campaign orchestration and does not alter this single-run trainer's state or
resume semantics.
