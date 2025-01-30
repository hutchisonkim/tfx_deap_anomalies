from tfx.types import Channel, ComponentSpec, standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter
from tfx.dsl.components.base import BaseComponent, executor_spec
from tfx.dsl.components.base.base_executor import BaseExecutor
from deap_tuner.deap_executor import DEAPExecutor

class DEAPTunerSpec(ComponentSpec):
    """ComponentSpec for the DEAPTuner."""
    PARAMETERS = {
        'population_size': ExecutionParameter(type=int),
        'generations': ExecutionParameter(type=int),
        'crossover_prob': ExecutionParameter(type=float),
        'mutation_prob': ExecutionParameter(type=float),
    }
    INPUTS = {
        'train_data': ChannelParameter(type=standard_artifacts.Examples),
        'test_data': ChannelParameter(type=standard_artifacts.Examples),
        'target_data': ChannelParameter(type=standard_artifacts.Examples),
    }
    OUTPUTS = {
        'best_hyperparameters': ChannelParameter(type=standard_artifacts.HyperParameters),
    }

class DEAPTuner(BaseComponent):
    """Custom TFX Tuner using DEAP for hyperparameter optimization."""

    SPEC_CLASS = DEAPTunerSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(DEAPExecutor)

    def __init__(
        self,
        train_data: Channel,
        test_data: Channel,
        target_data: Channel,
        population_size: int = 50,
        generations: int = 20,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
    ):
        """Constructs the DEAPTuner component."""
        best_hyperparameters = Channel(type=standard_artifacts.HyperParameters)

        spec = DEAPTunerSpec(
            train_data=train_data,
            test_data=test_data,
            target_data=target_data,
            population_size=population_size,
            generations=generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            best_hyperparameters=best_hyperparameters,
        )

        super().__init__(spec=spec)
