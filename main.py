# -*- coding: utf-8 -*-

##Import Azure SDK modules
import azureml.core
from azureml.core import Workspace ,Experiment
from azureml.core.model import Model
from azureml.core import ScriptRunConfig
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule
from azureml.core import Run



# plt.rcParams["figure.figsize"] = (10,10)
import warnings

warnings.filterwarnings("ignore")

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
exp=Experiment(workspace=ws,name='loanappvexpv03')

#Start logging metrics
import datetime

run=exp.start_logging()
run.log("Experiement start time",str(datetime.datetime.now()))




# =============================================================================
# #Create Azure ML Experiment
# exp=Experiment(workspace=ws,name='loanappvexpv03')
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





from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Environment


myenv = Environment.get(workspace=ws, name="newenv")

run_config = RunConfiguration()
run_config.environment = myenv



# =============================================================================
# aml_run_config = RunConfiguration()
# # `compute_target` as defined in "Azure Machine Learning compute" section above
# aml_run_config.target = compute_target
# 
# 
# try:
#     curated_environment = Environment.get(workspace=ws, name="LoanApprovalEnv")
# except : 
#     # From a Conda specification file
#      myenv = Environment.from_conda_specification(name = "LoanApprovalEnv",
#                                                  file_path = "./env.yml") 
#      myenv.register(workspace=ws)
#      curated_environment=myenv
#      
# aml_run_config.environment = curated_environment     
#         
# =============================================================================

environments = Environment.list(workspace=ws)

for environment_name in environments:
    print(environment_name)


# Get the blob storage associated with the workspace
from azureml.core import  Datastore

def_blob_store = Datastore(ws, "workspaceblobstore")


from azureml.pipeline.core import PipelineData
processed_data = PipelineData("processed_data", datastore=def_blob_store)
trained_model = PipelineData("trained_model",datastore=def_blob_store)

from azureml.pipeline.steps import PythonScriptStep
dataprep_source_dir = "./"
entry_point = "./Train/prepare.py"


# `output_data1`, `compute_target`, `aml_run_config` as defined above
data_prep_step = PythonScriptStep(
    script_name=entry_point,
    source_directory=dataprep_source_dir,
    arguments=["--output", processed_data],
    outputs=[processed_data],
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=False
)

from azureml.pipeline.steps import PythonScriptStep
dataprep_source_dir = "./"
entry_point = "./Train/model.py"

# `output_data1`, `compute_target`, `aml_run_config` as defined above
model_train_step = PythonScriptStep(
    script_name=entry_point,
    source_directory=dataprep_source_dir,
    arguments=["--input", processed_data,"--output",trained_model],
    inputs=[processed_data],
    outputs=[trained_model],
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=False
)


from azureml.pipeline.core import Pipeline, PublishedPipeline

steps = [data_prep_step, model_train_step]


#Design Pipelines
pipeline = Pipeline(workspace=ws, steps=steps)
pipeline.validate()

from azureml.core import Experiment

# Submit the pipeline to be run
pipeline_run1 = Experiment(ws, 'TrainModelPipeline').submit(pipeline)
pipeline_run1.wait_for_completion()



# =============================================================================
# ##### Publish Pipelines #######
# 
# from azureml.pipeline.core.graph import PipelineParameter
# 
# published_pipeline1 = pipeline_run1.publish_pipeline(
#      name="My_Published_Pipelinev01",
#      description="My Published Pipeline Description",
#      version="1.0")
# 
# 
# 
# 
# #### List out published pipelines ####
# published_pipelines = PublishedPipeline.list(ws)
# for published_pipeline in  published_pipelines:
#     print(f"{published_pipeline.name},'{published_pipeline.id}'")
# 
# 
# pipeline_id='103b001e-4b62-4f85-b213-ce50a6c050b7'
# exp='ScheduledPipeline'
# ################ Trigger Pipeline on scheduled time ###########
# from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule
# 
# recurrence = ScheduleRecurrence(frequency="Month", interval=1)
# recurring_schedule = Schedule.create(ws, name="MyRecurringSchedule", 
#                             description="Based on time",
#                             pipeline_id=pipeline_id, 
#                             experiment_name=exp, 
#                             recurrence=recurrence)
# 
# 
# # Disable schedule jobs
# 
# Schedule.list(ws)[0].disable()
# 
# =============================================================================

# =============================================================================
# # Get the pipeline by using its ID from Azure Machine Learning studio
# p = PublishedPipeline.get(ws, id=pipeline_id)
# p.disable()
# =============================================================================





