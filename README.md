# LightTS

Code for the paper LightTS: Lightweight Time Series Classification with Adaptive Ensemble Distillation (SIGMOD 2023)

How to run the model:
 * Execute [main.py](main.py) specifying the model parameters, for example:
 `python main.py --dataset "NonInvasiveFetalECGThorax2" --evaluation "student" --teacher_type "Inception" --teachers 10 --evaluation "lighths" --bit1 4 --bit2 4 --bit3 4`
 * The complete list of parameters in [main.py](main.py).
 * Data is managed in [data.py](./utils/data.py). It follows the UCR archive structure, so the data sets folders are expected in `./dataset/TimeSeriesClassification/`
 * Results will be inserted in a database, connections are managed in [util.py](./utils/util.py).

# Citation

If you use the code, please cite the following paper:

<pre>  
@article{abs-2302.12721,
  author    = {David Campos and Miao Zhang and Bin Yang and Tung Kieu and Chenjuan Guo 
               and Christian S. Jensen},
  title     = {{LightTS: Lightweight Time Series Classification with Adaptive Ensemble}},
  journal   = {CoRR},
  volume    = {abs/2302.12721},
  year      = {2023},
  url       = {https://doi.org/10.48550/arXiv.2302.12721},
}
</pre> 