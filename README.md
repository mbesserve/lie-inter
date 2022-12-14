# Learning soft interventions in complex equilibrium systems
Michel Besserve & Bernhard Schölkopf

Max Planck Institute for Intelligent Systems, Tübingen, Germany.

[Paper at UAI 2022](https://openreview.net/forum?id=HrBgldLo9ec)


## Code description
### Toy experiments
Code for the toy optimization experiments is provided in the code subfolder:
  -  leontiefPriceinvarUAI.py reproduces the invariant intervention experiment of paragraph “Control of rebound effects” in
main text Sec. 5,
  -  leontiefCompartv2.py reproduces the compartmentalized intervention experiment of paragraph “Compartmentalized
intervention design” in main text Sec. 5.

### Semi-synthetic experiments
They are based on the [ExioBase3](https://www.exiobase.eu/index.php/9-blog/31-now-available-exiobase2) dataset, which can be processed throught the [pymrio](https://pymrio.readthedocs.io) library. 


## How to run

First, clone this repository  
```bash
# clone ima_vae   
git clone https://github.com/mbesserve/lie-inter
```
Then install required libraries
```bash
cd lie-inter
pip install -r requirements.txt
```
