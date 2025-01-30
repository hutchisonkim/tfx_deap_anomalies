import datetime
import os
from typing import List

import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing

import sys

# Add the src directory to the Python module path
sys.path.append(os.path.join(os.environ['HOME'], 'src'))

# Import the custom DEAP tuner and executor
from deap_tfx.deap_tuner_component import DEAPTuner
from deap_tuner.deap_executor import DEAPExecutor

# Fraud detection pipeline configurations
_pipeline_name = 'fraud_detection_pipeline'

# Define the root and data paths
_tfx_root = os.path.join(os.environ['HOME'], 'fraud_detection_tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)

# Input and output paths for the fraud detection dataset
_data_root = os.path.join(_tfx_root, 'data', 'ieee_cis_fraud_detection')
_serving_model_dir = os.path.join(_tfx_root, 'serving_model', _pipeline_name)

# Path to the fraud detection utils module
_module_file = os.path.join('src', 'fraud_detection_tfx', 'fraud_detection_utils.py')
import fraud_detection_tfx.fraud_detection_utils

_LABEL_KEY = 'isFraud'

# Beam pipeline arguments
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    '--direct_num_workers=0',
]

# Airflow-specific configs for scheduling
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2025, 1, 1),
}

def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str, beam_pipeline_args: List[str]) -> pipeline.Pipeline:
    """Implements the fraud detection pipeline using TFX."""

    # Parametrize data root so it can be replaced on runtime
    data_root_runtime = data_types.RuntimeParameter('data_root', ptype=str, default=data_root)

    # Brings raw data into the pipeline (CSV input in this case)
    example_gen = CsvExampleGen(input_base=data_root_runtime)

    # Computes statistics over data for visualization and example validation
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Generates schema based on statistics files
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=False)

    # Performs example validation based on computed statistics and schema
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Perform data transformations such as feature engineering and preprocessing
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file
    )

    # Set up the custom DEAPOptimizer tuner
    tuner = DEAPTuner(
        train_data=transform.outputs['transformed_examples'],
        test_data=transform.outputs['test_examples'],
        target_data=transform.outputs['target_examples'],
    )

    # Train the model using a user-defined function (trainer)
    trainer = Trainer(
        module_file=module_file, 
        transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000),
        hyperparameters=tuner.outputs['best_hyperparameters'],
    )

    # Model resolver to get the latest blessed model for validation
    model_resolver = resolver.Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')


    # TFMA evaluation configuration for model performance analysis
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(signature_name='eval')],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=[_LABEL_KEY])  # Slicing by fraud labels
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                thresholds={
                    'accuracy': tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(lower_bound={'value': 0.7}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER, absolute={'value': -1e-10}
                        )
                    )
                }
            )
        ]
    )
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    # Push the trained model if validation passes
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=serving_model_dir)
        )
    )

    # Return the TFX pipeline
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, statistics_gen, schema_gen, example_validator, transform,
            tuner, trainer, model_resolver, evaluator, pusher
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
        beam_pipeline_args=beam_pipeline_args
    )


# Airflow DAG setup for execution
_metadata_path = 'mysql://root:rootpassword@mysql:3306/metadata_db'
DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
    _create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        data_root=_data_root,
        module_file=_module_file,
        serving_model_dir=_serving_model_dir,
        metadata_path=_metadata_path,
        beam_pipeline_args=_beam_pipeline_args
    )
)
