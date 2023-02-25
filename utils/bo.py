from ax import *
from ax.runners.synthetic import SyntheticRunner
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from botorch.models.gp_regression import SingleTaskGP
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
from plotly.offline import plot
from sktime.datatypes._panel._convert import from_2d_array_to_nested

class StudentBO():
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.init_teachers = self.config.teachers

    def __call__(self,params):
        self.config.bit1 = params["bit_1"]
        self.config.bit2 = params["bit_2"]
        self.config.bit3 = params["bit_3"]
        self.config.layer1 = params["layers_1"]
        self.config.layer2 = params["layers_2"]
        self.config.layer3 = params["layers_3"]
        max_accuracy = pivot_accuracy = 0

        if self.config.specific_teachers:
            teachers = config.list_teachers
        else:    
            teachers = [i for i in range(0,self.init_teachers)]
        max_accuracy, teacher_weights = RunStudent(self.model, self.config, teachers)

        if self.config.leaving_out:
            max_accuracy = recursive_accuracy(self.model, self.config, max_accuracy, teachers)

        if self.config.leaving_weights:
            max_accuracy = recursive_weight(self.model, self.config, teacher_weights)

        return max_accuracy

def build_experiment(search_space,optimization_config):
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment

def initialize_experiment(experiment,initialization):
    sobol = Models.SOBOL(search_space=experiment.search_space, seed=1234)

    for _ in range(initialization):
        trial = experiment.new_trial(sobol.gen(1))
        trial.run()
        trial.mark_completed()

    return experiment.fetch_data()


class MetricAccuracy(Metric):
    def fetch_trial_data(self, trial):  
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": student_bo(params),
                "sem": 0,
            })
        return Data(df=pd.DataFrame.from_records(records))
    
class MetricCost(Metric):
    def fetch_trial_data(self, trial):  
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            bit_cost = params["layers_1"] * params["bit_1"] + params["layers_2"] * params["bit_2"] + params["layers_3"] * params["bit_3"]
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": bit_cost,
                "sem": 0,
            })
        return Data(df=pd.DataFrame.from_records(records))
    
def BayesianOptimization(config):
    config.layer1 = config.layer2 = config.layer3 = 3
    model_s = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                   bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                   num_pred_classes=config.num_classes,config=config)
    model_s = model_s.to(config.device)
    student_bo = StudentBO(model_s, config)
    
    bit_1=ChoiceParameter(name="bit_1", values=[5,6,7], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    bit_2=ChoiceParameter(name="bit_2", values=[5,6,7], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    bit_3=ChoiceParameter(name="bit_3", values=[5,6,7], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    layers_1=ChoiceParameter(name="layers_1", values=[3,4], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    layers_2=ChoiceParameter(name="layers_2", values=[3,4], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    layers_3=ChoiceParameter(name="layers_3", values=[3,4], parameter_type=ParameterType.INT,sort_values=True,is_ordered=True)
    #layers_1=FixedParameter(name="layers_1", value=3, parameter_type=ParameterType.INT)
    #layers_2=FixedParameter(name="layers_2", value=3, parameter_type=ParameterType.INT)
    #layers_3=FixedParameter(name="layers_3", value=3, parameter_type=ParameterType.INT)


    search_space = SearchSpace(parameters=[bit_1, bit_2, bit_3, layers_1, layers_2, layers_3])
    
    metric_accuracy = GenericNoisyFunctionMetric("accuracy", f=student_bo, noise_sd=0.0, lower_is_better=False)

    #metric_accuracy2 = MetricAccuracy(name="accuracy2",lower_is_better=False)
    metric_cost = MetricCost(name="cost",lower_is_better=True)
    
    if config.evaluation == 'student_bo':
        objectives = MultiObjective(objectives=[Objective(metric=metric_accuracy), Objective(metric=metric_cost)])
        objective_thresholds = [
            ObjectiveThreshold(metric=metric_accuracy, bound=0.7, relative=False),
            ObjectiveThreshold(metric=metric_cost, bound=45, relative=False),
        ]

        optimization_config = MultiObjectiveOptimizationConfig(
            objective=objectives,
            objective_thresholds=objective_thresholds,
        )

        bo_experiment = build_experiment(search_space,optimization_config)
        bo_data = initialize_experiment(bo_experiment,config.bo_init)

        bo_model = None
        for i in range(config.bo_steps):
            bo_model = Models.MOO_MODULAR(
                experiment=bo_experiment, data=bo_data,
                surrogate=ListSurrogate(
                botorch_submodel_class_per_outcome={"accuracy": SingleTaskGP, "cost": SingleTaskGP,},
                submodel_options_per_outcome={"accuracy": {}, "cost": {}},))

            generator_run = bo_model.gen(1)
            params = generator_run.arms[0].parameters

            trial = bo_experiment.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()
            bo_data = Data.from_multiple_data([bo_data, trial.fetch_data()])
            
            exp_df = exp_to_df(bo_experiment)

            outcomes = np.array(exp_to_df(bo_experiment)[['accuracy', 'cost']], dtype=np.double)

            frontier = compute_posterior_pareto_frontier(
                experiment=bo_experiment,
                data=bo_experiment.fetch_data(),
                primary_objective=metric_accuracy,
                secondary_objective=metric_cost,
                absolute_metrics=["accuracy", "cost"],
                num_points=config.bo_init + config.bo_steps,
            )

            plot(plot_pareto_frontier(frontier, CI_level=0.90).data, filename=config.experiment+'_'+str(config.pid)+'_.html')
    
    elif config.evaluation == 'student_bo_simple':
        bo_experiment = SimpleExperiment(search_space=search_space,evaluation_function=student_bo)
        bo_experiment.runner = SyntheticRunner()
        config.bo_status = 'Random'
        bo_data = initialize_experiment(bo_experiment,config.bo_init)
        config.bo_status = 'Optimized'
        bo_model = None
        for i in range(config.bo_steps):
            bo_model = Models.BOTORCH(experiment=bo_experiment, data=bo_data)

            generator_run = bo_model.gen(1)
            params = generator_run.arms[0].parameters

            trial = bo_experiment.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()
            bo_data = Data.from_multiple_data([bo_data, trial.fetch_data()])

            exp_df = exp_to_df(bo_experiment)