# Project 2: RNA EM Algorithms

## Build instructions
I decided to change the arguments from "in" and "out" to "input" and "output" since "in" is a keyword in python. I tried using a shebang to specify that this program ran in python so that you didn't have to specify python on the command line however that continuously failed to work. 
The user can optionally add the --eqc flag that will run the equivalence class model EM algorithm. Not specifying that flag will run the full model EM algorithm. 
python squant.py --input INPUT --output OUTPUT [--eqc]
