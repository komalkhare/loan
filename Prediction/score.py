# -*- coding: utf-8 -*-

##Import Azure SDK modules
import azureml.core
from azureml.core import Workspace ,Experiment ,Environment ,RunConfiguration
from azureml.core.model import Model
from azureml.core import ScriptRunConfig
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule
from azureml.pipeline.steps import PythonScriptStep

# plt.rcParams["figure.figsize"] = (10,10)

# =============================================================================
# #Create Azure ML workspace
# ws=Workspace.create(name='loanappvModel',
#                    subscription_id='f89c6bbf-835b-4d11-8af1-9326831974d0',
#                     resource_group='Aspire_Private_RG',
#                     create_resource_group=True,
#                     location='centralus'
#                    
#                    
#                    )
# 
# #Write configuration to local file
# ws.write_config()
# =============================================================================


#Access from local system
ws=Workspace.from_config()
ws.get_details()


#Create Azure ML Experiment
exp=Experiment(workspace=ws,name='ScoreDataPipeline')

#Start logging metrics
import datetime

run=exp.start_logging()
run.log("Experiement start time",str(datetime.datetime.now()))
# =============================================================================
# 
# 
# #Cancel stuck runs
# for run in exp.get_runs():
#     if run.get_status()=='Running':
#         run.cancel()
#         print('Cancelled run with Id : ',run.id)
#     else:
#         pass
# =============================================================================


#Get portal URL
print(run.get_portal_url())

# Create Compute
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


compute_name = "aml-compute"
vm_size = "Standard_D2_v3"
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('Found compute target: ' + compute_name)
else:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,  # STANDARD_NC6 is GPU-enabled
                                                                min_nodes=0,
                                                                max_nodes=2,
                                                                vm_priority='dedicated'
                                                                )
    # create the compute target
    compute_target = ComputeTarget.create(ws, 
                                            compute_name, 
                                            provisioning_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, 
                                       min_node_count=None, 
                                       timeout_in_minutes=20)

    # For a more detailed view of current cluster status, use the 'status' property
    print(compute_target.status.serialize())

# =============================================================================
# 
# from azureml.core.runconfig import RunConfiguration
# from azureml.core.conda_dependencies import CondaDependencies
# from azureml.core import Environment
# 
# 
# Newenv = CondaDependencies("./configs/env.yml")
# run_config = RunConfiguration(conda_dependencies=Newenv)
# myenv = Environment.from_conda_specification(name = "newevnv01",
#                                                   file_path = "./configs/env.yml") 
# run_config.environment = myenv
# 
# print("===== Conda dependencies have been loaded======")
# # Creates the environment inside a Docker container.
# run_config.environment.docker.enabled = True
# run_config.environment.docker.base_image=None
# print("===== Docker dependencies have been set to True======")
# run_config.environment.docker.base_dockerfile = "./doc.Dockerfile"
# print("===== Docker dependencies file have been loaded=====")
# =============================================================================


# =============================================================================
# 
# aml_run_config.environment = curated_environment    
# print("===== Conda dependencies have been loaded======")
# # Creates the environment inside a Docker container.
# aml_run_config.environment.docker.enabled = True
# aml_run_config.environment.docker.base_image=None
# print("===== Docker dependencies have been set to True======")
# aml_run_config.environment.docker.base_dockerfile = "./doc.Dockerfile"
# print("===== Docker dependencies file have been loaded=====") 
# 
# =============================================================================
# =============================================================================
# environments = Environment.list(workspace=ws)
# 
# for environment_name in environments:
#     print(environment_name)
# 
# =============================================================================

# =============================================================================
# # Get the image configuration details
# aml_run_config.environment.get_image_details(workspace=ws)
# 
# ##  Build the environment locally . So one can check if everything is good locally in lieu of running whole pipeline.
# build = run_config.environment.build(workspace=ws)
# build.wait_for_completion(show_output=True)
# 
# =============================================================================

myenv = Environment.get(workspace=ws, name="newenv")

run_config = RunConfiguration()
run_config.environment = myenv

from azureml.pipeline.steps import PythonScriptStep
dataprep_source_dir = "./"
entry_point = "./Prediction/prepareScore.py"


#compute_target='local'

# `output_data1`, `compute_target`, `aml_run_config` as defined above
predictScoreData = PythonScriptStep(
    script_name=entry_point,
    source_directory=dataprep_source_dir,
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=False
)


from azureml.pipeline.core import Pipeline, PublishedPipeline

steps = [predictScoreData]

#Design Pipelines
pipeline = Pipeline(workspace=ws, steps=steps)
pipeline.validate()

from azureml.core import Experiment

# Submit the pipeline to be run
pipeline_run1 = Experiment(ws, 'ScoreDataPipeline').submit(pipeline)
pipeline_run1.wait_for_completion()


