"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import (FrameworkProcessor, ProcessingInput,
                                  ProcessingOutput, ScriptProcessor)
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.serializers import JSONSerializer
from sagemaker.sklearn import SKLearn, SKLearnProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import (ConditionGreaterThanOrEqualTo,
                                           ConditionLessThanOrEqualTo)
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="IntelImageClassifierModelGroup",
    pipeline_name="PyTorchLightningKaggleIntelImageClassifier",
    base_job_prefix="IntelImage",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    dvc_repo_url = ParameterString(
        name="DVCRepoURL",
        default_value="codecommit::ap-south-1://sagemaker-intel-image-classification",
    )
    dvc_branch = ParameterString(
        name="DVCBranch", default_value="pipeline-processed-dataset"
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_dataset = ParameterString(
        name="InputDatasetZip",
        default_value="s3://sagemaker-ap-south-1-294495367161/labelled_dataset/",
    )

    base_job_name = base_job_prefix

    # PREPROCESS STEP

    sklearn_processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="0.23-1",
        # instance_type="ml.t3.medium",
        instance_type="ml.t3.large",
        image_uri="294495367161.dkr.ecr.ap-south-1.amazonaws.com/sagemaker:latest",
        # instance_type='local',
        instance_count=1,
        base_job_name=f"{base_job_name}/preprocess-dataset",
        sagemaker_session=pipeline_session,
        # sagemaker_session=local_pipeline_session,
        role=role,
        env={
            "DVC_REPO_URL": dvc_repo_url,
            "DVC_BRANCH": dvc_branch,
            "GIT_USER": "Vivek Chaudhary",
            "GIT_EMAIL": "vivek.experiotech@gmail.com",
        },
    )

    processing_step_args = sklearn_processor.run(
        code="preprocess.py",
        source_dir=BASE_DIR,
        # dependencies="sagemaker-flower-pipeline/requirements.txt",
        inputs=[
            ProcessingInput(
                input_name="data",
                source=input_dataset,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train", source="/opt/ml/processing/dataset/train"
            ),
            ProcessingOutput(
                output_name="val", source="/opt/ml/processing/dataset/val"
            ),
            ProcessingOutput(
                output_name="test", source="/opt/ml/processing/dataset/test"
            ),
        ],
    )

    step_process = ProcessingStep(
        name="PreprocessDataset",
        step_args=processing_step_args,
    )

    # TRAIN STEP
    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{default_bucket}/logs/sagemaker_intel_image_classification",
        container_local_output_path="/opt/ml/output/tensorboard",
    )

    pt_estimator = PyTorch(
        image_uri="294495367161.dkr.ecr.ap-south-1.amazonaws.com/sagemaker:latest",
        base_job_name=f"{base_job_name}/training_intel_image",
        source_dir=BASE_DIR,
        entry_point="train.py",
        sagemaker_session=pipeline_session,
        role=role,
        py_version="py38",
        framework_version="1.11.0",
        instance_count=1,
        instance_type="ml.c5.4xlarge",
        tensorboard_output_config=tensorboard_output_config,
        use_spot_instances=True,
        max_wait=15500,
        max_run=15200,
        environment={
            "GIT_USER": "Vivek Chaudhary",
            "GIT_EMAIL": "vivek.experiotech@gmail.com",
        },
    )

    estimator_step_args = pt_estimator.fit(
        {
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
            ),
            "test": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
            ),
            "val": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "val"
                ].S3Output.S3Uri,
            ),
        }
    )

    step_train = TrainingStep(
        name="TrainClassifier",
        step_args=estimator_step_args,
    )

    # EVAL STEP

    pytorch_processor = PyTorchProcessor(
        image_uri="294495367161.dkr.ecr.ap-south-1.amazonaws.com/sagemaker:latest",
        framework_version="1.11.0",
        py_version="py38",
        role=role,
        sagemaker_session=pipeline_session,
        instance_type="ml.t3.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_name}/eval-intel_image-classifier-model",
    )

    eval_step_args = pytorch_processor.run(
        code="eval.py",
        source_dir=BASE_DIR,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation"
            ),
        ],
    )

    evaluation_report = PropertyFile(
        name="IntelImageClassifierEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="ModelEvaluation",
        step_args=eval_step_args,
        property_files=[evaluation_report],
    )

    # MODEL REGISTER STEP

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ]
            ),
            content_type="application/json",
        )
    )

    model = PyTorchModel(
        entry_point="infer.py",
        source_dir=BASE_DIR,
        sagemaker_session=pipeline_session,
        role=role,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        # model_data="s3://sagemaker-ap-south-1-006547668672/training-flower-pipeline-2022-12-07-03-20-21-157/output/model.tar.gz",
        framework_version="1.11.0",
        py_version="py38",
    )

    model_step_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m4.xlarge"],
        model_package_group_name=model_package_group_name,
        # approval_status=model_approval_status,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    step_register = ModelStep(
        name="RegisterModel",
        step_args=model_step_args,
    )

    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.accuracy.value",
        ),
        right=0.6,
    )

    step_cond = ConditionStep(
        name="CheckEvaluationAccuracy",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[dvc_repo_url, dvc_branch, input_dataset, model_approval_status],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )

    return pipeline
