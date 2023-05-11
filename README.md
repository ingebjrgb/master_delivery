# master_recommendation_blur

## Project Description
This repository includes an implemention of recommender list obfuscation based on replacement. The obfuscation attempts to lower the privacy threats evident in recommender lists. 

## Installation and Running
The main file in this repostory is the Obfuscation.py-file that implements the obfuscations. To test the privacy preservation of the obfuscations, the jupyter notebook inference_attack.ipynb is used. This repository does not include any of the data used, however this data can be retrieved from https://movielens.org/ .

### Obfuscation.py
Runs the obfuscations. For gender based removal strategies, a file filled with the gender typical movies must be present. These gender typical movie lists are generated by running female_or_male.py.
For kfn-based obfuscations, the kfn recs must be generated. These recommendations are generated by running kfn.py.

### inference_attack.ipynb 
Implements the inference attacks and measures their performance. 


