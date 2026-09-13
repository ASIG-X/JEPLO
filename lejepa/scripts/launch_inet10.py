import stable_pretraining as spt

TEMPLATE = """
HYDRA_FULL_ERROR=1 python scripts/je.py --config-dir scripts/configs --config-name base +accelerator=single_gpu_sc ++bstat_name="epps_pulley" ++bstat_num_slices=1000 ++dataset_name="inet10" ++max_epochs=400 ++batch_size=512 ++bstat_lambda=0.01,0.02,0.05,0.1 ++embedding_dim=512 ++projector_dim=512 ++projector_arch="MLP" ++lr=3e-3,1e-4 ++weight_decay=3e-2,1e-5 ++num_nodes=1 ++hydra.launcher.gpus_per_node=2 ++teacher_student=false ++resolution=224 ++n_views=8 ++random_erasing_p=0 ++drop_path_rate=0 ++multi_crop=false ++predictor=false ++autostop=true ++wandb_project=null 
"""

names = []
count = 0
for name, value in spt.static.TIMM_PARAMETERS.items():
    if value < 20000000 and "." not in name:
        count += 1
        names.append(f'"{name}"')

print(f"RUN\n\n{TEMPLATE} ++backbone={','.join(names)}")
