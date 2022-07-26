# Parameters selection

---

To choose the dynamics model, select `training_params.inference_algo` from:
* Causal Dynamics Learning: `"cmi"`
* Monolithic: `"mlp"`
* Regularization: `"reg"`
* Modular: `"reg"` and change `inference_params.reg_params.use_mask` to `false`
* Graph Neural Network: `"gnn"`
* Neural Production System: `"nps"`

To choose manipulation tasks, select `env_params.env_name` from `"CausalReach"`, `"CausalPush"`, `"CausalPick"`, `"CausalStack"`.