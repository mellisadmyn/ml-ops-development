"""
Modul ini berisi fungsi untuk menginisialisasi semua komponen TFX yang
digunakan dalam pipeline machine learning end-to-end.
"""

import os
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Tuner,
    Evaluator,
    Pusher,
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)

# Fungsi untuk melakukan inisialisasi komponen
def init_components(config):
    """
    Mengembalikan komponen TFX untuk pipeline.

    Args:
        config (dict): Dictionary konfigurasi dengan path dan pengaturan.

    Returns:
        tuple: Komponen pipeline TFX.
    """

    # Membagi dataset dengan perbandingan 80% untuk training dan 20% untuk evaluasi
    output = example_gen_pb2.Output(    # pylint: disable=no-member
        split_config=example_gen_pb2.SplitConfig(   # pylint: disable=no-member
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),    # pylint: disable=no-member
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),     # pylint: disable=no-member
            ]
        )
    )

    # Komponen example gen
    example_gen = CsvExampleGen(
        input_base=config["DATA_ROOT"],
        output_config=output,
    )

    # Komponen statistics gen
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # Komponen schema gen
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )

    # Komponen example validator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # Komponen transform menggunakan modul transform.py
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath(config["transform_module"]),
    )

    # Komponen tuner menggunakan modul tuner.py
    tuner = Tuner(
        module_file=os.path.abspath(config["tuner_module"]),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(  # pylint: disable=no-member
            splits=["train"],
            num_steps=config["training_steps"],
        ),
        eval_args=trainer_pb2.EvalArgs(  # pylint: disable=no-member
            splits=["eval"],
            num_steps=config["eval_steps"],
        ),
    )

    # Komponen trainer menggunakan modul trainer.py
    trainer = Trainer(
        module_file=os.path.abspath(config["training_module"]),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(  # pylint: disable=no-member
            splits=["train"],
            num_steps=config["training_steps"],
        ),
        eval_args=trainer_pb2.EvalArgs(  # pylint: disable=no-member
            splits=["eval"],
            num_steps=config["eval_steps"],
        ),
    )

    # Komponen model resolver
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolver")

    # Konfigurasi metrik evaluasi
    metrics_specs = [
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name="AUC"),
                tfma.MetricConfig(class_name="Precision"),
                tfma.MetricConfig(class_name="Recall"),
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(
                    class_name="BinaryAccuracy",
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": 0.8}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value": 0.0001},
                        ),
                    ),
                ),
            ]
        )
    ]

    # Konfigurasi evaluasi
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="Attrition")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=metrics_specs,
    )

    # Komponen evaluator
    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    # Komponen pusher
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(  # pylint: disable=no-member
            filesystem=pusher_pb2.PushDestination.Filesystem(  # pylint: disable=no-member
                base_directory=config["serving_model_dir"]
            )
        ),
    )

    # Mengembalikan semua komponen pipeline
    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )
