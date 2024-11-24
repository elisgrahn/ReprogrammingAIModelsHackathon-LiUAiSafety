# Feature Tuning versus Prompting for Ambiguous Questions
## About
This repository contains LiU AI Safety's submission for the [Reprogramming AI Models Hackathon](https://www.apartresearch.com/event/reprogramming-ai-models-hackathon).

The goal of the hackathon is to use the Goodfire SDK to investigate LLM features and how they can be used for AI safety and related tasks.

We chose to investigate the effect of feature tuning versus prompting on the performance of an LLM on ambiguous questions. This is a proxy for the model's usage of cognitive ease and the ambiguity effect, human psychological biases which are exacerbated during LLM training. We find that feature tuning is on par with prompting in answer quality, but that combining the two leads to the best performance.

For further details, see our [research paper](LiU_AI_Safety_Reprogramming_AI.pdf).

## Setup
Create a virtual environment and install the Goodfire package.
```
python3 -m venv venv
source venv/bin/activate
pip install goodfire
```

### Goodfire key
Create a file called ```goodfire.key``` and paste your Goodfire API key in there, it will not be tracked by git. You can get your key [here](https://platform.goodfire.ai/organization/settings/api-keys).

## Usage
You can run the experiments with the following command:
```
python ambiguity.py
```
Note that depending on server load, there might be minor delay.

The questions and answers for each variant (baseline, feature tuning, hidden prompt, both) will be printed to the console.