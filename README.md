# LightTS

Code for the paper LightTS: Lightweight Time Series Classification with Adaptive Ensemble Distillation (SIGMOD 2023)

How to run the model:
 * Create the conda environment: `conda env create -f environment.yaml`
 * Activate the environment: `conda activate lightts`
 * Execute [main.py](main.py) to compute the teacher models, for example:
 `python main.py --dataset "Adiac" --experiment "teacher" --teacher_type "Inception" --teachers 10`
 * Execute [main.py](main.py) specifying the lightweight model requirements, for example:
`python main.py --dataset "Adiac" --experiment "student" --teacher_type "Inception" --teachers 10 --evaluation "lightts" --bit1 4 --bit2 4 --bit3 4`
 * The complete list of parameters is in [main.py](main.py).
 * Results are printed in the prompt for illustration. Reported results are managed in a externally in a database, as it is detailed in [util.py](./utils/util.py).

# Citation

If you use the code, please cite the following paper:

<pre>  
@article{pacmmod/0002Z0KGJ23,
  author       = {David Campos and
                  Miao Zhang and
                  Bin Yang and
                  Tung Kieu and
                  Chenjuan Guo and
                  Christian S. Jensen},
  title        = {LightTS: Lightweight Time Series Classification with Adaptive Ensemble
                  Distillation},
  journal      = {Proc. {ACM} Manag. Data},
  volume       = {1},
  number       = {2},
  pages        = {171:1--171:27},
  year         = {2023},
  url          = {https://doi.org/10.1145/3589316},
  doi          = {10.1145/3589316},
}
</pre> 
