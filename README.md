# loop-estimator
Policy evaluation in the infinite-horizon discounted reward setting in RL. This is the code associated with the following publication https://arxiv.org/abs/2002.06299
```
Dai, Falcon Z and Walter, Matthew R. Loop Estimator for Discounted Values in Markov Reward Processes. Proceedings of AAAI. 2021.
```

## File organization
- `demo.ipynb` is the python notebook of experiments including the plots in the main paper.
- `estimate.py` contains the estimators for state values, namely `co_loop` for the loop estimator, `co_td_k` for TD(k) estimator, `co_model_based` for the model-based estimator. See their definitions in the paper. Their implementations extensively exploit co-routines, i.e., `yield` statements, to enhance both readability and efficiency.
- `mrp.py` contains the definition of Markov reward processes and, in particular, the definition of RiverSwim.
- `mc.py` contains some utility functions for Markov chains.
- `*.npy` are pre-computed state value estimates from the different estimators (used in generating the plots in the main paper).

## Dependency
- `python 3.x`
- `jupyter`. Install by `pip3 install jupyter`

## Replication
To replicate the experimental results in the paper:
- Start the jupyter notebook server at the project root
```bash
jupyter notebook
```
- Select the notebook `demo.ipynb`
- Follow the comments within. Optionally load the pre-computed estimates instead of re-computing them.

## Reference
Please cite our work if you find this repo or the associated [paper](https://arxiv.org/abs/2002.06299) useful.

```bibtex
@inproceedings{dai-walter-2021-loop,
    title = "Loop Estimator for Discounted Values in Markov Reward Processes",
    author = "Dai, Falcon Z and Walter, Matthew R",
    booktitle = "Proceedings of Association for the Advancement of Artificial Intelligence Conference",
    month = feb,
    year = "2021",
    publisher = "Association for the Advancement of Artificial Intelligence"
  }
```

## Author
Falcon Dai (me@falcond.ai)
