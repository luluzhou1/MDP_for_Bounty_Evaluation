# MDP Model for Bounty Evaluation

## Description
This is the artifact of the paper: [CrudiTEE: A Stick-and-Carrot Approach to Building Trustworthy Cryptocurrency Wallets with TEEs
](http://arxiv.org/abs/2407.16473).

## Instructions for reproducing the result in the Paper
### Step 1: 
Under the ``./MDP_for_Bounty_Evaluation`` dir, run
```
mkdir output
mkdir plot
```
### Step 2:
Set up python virtual environment:
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3:
Run the script under the ``./MDP_for_Bounty_Evaluation`` dir
```
python bounty_evaluation.py
```

The figure saved in ./plot is the Figure 4 in the paper.
