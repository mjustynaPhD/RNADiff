# %% [markdown]
# # Imports

# %%
import numpy as np
import importlib
import os
import torch
import GPUtil
import ml_collections
import time
import matplotlib.pyplot as plt
import tree
from plotly.subplots import make_subplots

from data import diffuser
from data import utils as du
from model import reverse_diffusion

from experiments import torch_train_diffusion
from analysis import plotting
from analysis import utils as au

torch.manual_seed(0)
np.random.seed(0)

# %%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
chosen_gpu = ''.join(
    [str(x) for x in GPUtil.getAvailable(order='memory')])
os.environ["CUDA_VISIBLE_DEVICES"] = chosen_gpu
print(chosen_gpu)

# %% [markdown]
# ### Set-up experiment

# %%
# Read ckpt
# ckpt_dir = 'ckpt/'
ckpt_dir = '/data/3d/RNADiff/torch_train_diffusion/18D_04M_2024Y_17h_14m_03s'
ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[4]).replace('.pth', '.pkl')

print(ckpt_path)
ckpt_pkl = du.read_pkl(ckpt_path)
ckpt_cfg = ckpt_pkl['cfg']
ckpt_state = ckpt_pkl['exp_state']

# %%
# Set-up experiment

data_setting = 'pdb'
cfg = torch_train_diffusion.get_config()
cfg = dict(cfg)
cfg['experiment'].update(ckpt_cfg.experiment)
cfg['experiment']['data_setting'] = data_setting
cfg['model'].update(ckpt_cfg.model)

# Pop unexpected model parameters
cfg['model'] = dict(cfg['model'])
# cfg['model'].pop('cross_prod_num_neighbors')
# cfg['model'].pop('inpainting_training')
cfg['model'].pop('num_heads')

cfg = ml_collections.ConfigDict(cfg)
cfg['data']['max_len'] = ckpt_cfg.data.max_len
cfg['data']['inpainting_training'] = False
cfg['data']['rmsd_filter'] = None
cfg['data']['monomer_only'] = True
print(cfg['data']['pdb_self_consistency_path'])


exp_cfg = cfg['experiment']
cfg['experiment']['batch_size'] = 4

exp = torch_train_diffusion.Experiment(cfg)
exp.model.load_state_dict(ckpt_state)

# %% [markdown]
# ### Sample

# %%
# Select number of samples and length of each sample
batch_size = 4

# %%
# Run sampling
sample_dir = 'model_samples'
os.makedirs(sample_dir, exist_ok=True)
noise_scale = 1.
LEN = 12
for num_res_sample in [LEN*8]:
    N = num_res_sample
    bb_mask = np.zeros((batch_size, N))
    bb_mask[:, :num_res_sample] = 1
    
    sampled_diffusion = exp.sample_reverse_diffusion(bb_mask)
    
    # Save reverse diffusion movies
    for b_idx in range(batch_size):
        save_path = f'{sample_dir}/len_{num_res_sample}_{b_idx}.pdb'
        au.write_prot_to_pdb(sampled_diffusion[b_idx][-1], save_path, no_indexing=True)

# %% [markdown]
# ### Visualize samples

# # %%
# ## Plot samples
# num_res = np.sum(bb_mask, axis=-1)
# nrows = int(np.sqrt(batch_size))
# ncols = nrows
# fig = make_subplots(
#     rows=nrows, cols=ncols,
#     specs=[[{'type': 'surface'}] * nrows]*ncols)

# # Take last time step
# last_sample = [x[-1] for x in sampled_diffusion]
# fig.update_layout(
#     title_text=f'Samples',
#     height=1000,
#     width=1000,
# )
# for i in range(nrows):
#     for j in range(ncols):
#         b_idx = i*nrows+j
#         sample_ij = last_sample[b_idx]
#         sample_bb_3d = plotting.create_scatter(
#             sample_ij, mode='lines+markers', marker_size=3,
#             opacity=1.0, name=f'Sample {i*nrows+j}: length={num_res[b_idx]}')
#         fig.add_trace(sample_bb_3d, row=i+1, col=j+1)
        
# fig.show()

# # %% [markdown]
# # # Conditional sampling test

# # %%
# from inpainting import motif_problems
# from inpainting import inpaint_experiment
# importlib.reload(motif_problems)

# # %%
# sample_dir = "inpaint_test_out/"
# os.makedirs(sample_dir, exist_ok=True)

# # %%
# # 6e6r test
# name = "6e6r"
# motif_start, motif_end = 10, 52
# pdb_name, target_len, motif_ca_xyz, full_ca_xyz_true, motif_idcs, inpainting_task_name = \
#     motif_problems.load_pdb_motif_problem(motif_start, motif_end, pdb_name=name, base_dir="./")

# # %%
# # Test with replacement method
# out = inpaint_experiment.run_inpainting(
#     exp, target_len, motif_ca_xyz, motif_idcs, exp.diffuser,
#     T=exp.cfg.experiment.T, N_samples_per_diffusion=4, inpainting_task_name="test", output_dir=sample_dir,
#     inpaint_method="replacement", num_save=4)

# # %%
# # Test with SMC-Diff
# out = inpaint_experiment.run_inpainting(
#     exp, target_len, motif_ca_xyz, motif_idcs, exp.diffuser,
#     T=exp.cfg.experiment.T, N_samples_per_diffusion=64, inpainting_task_name="test",
#     output_dir=sample_dir, inpaint_method="particle", num_save=4)

# # %%
# # 5trv scaffolding test
# name = "5trv"
# pad = 20
# motif_start, motif_end = 42, 62 # minimal
# motif_start, motif_end = motif_start-pad, motif_end+pad
# pdb_name, target_len, motif_ca_xyz, full_ca_xyz_true, motif_idcs, inpainting_task_name = \
#     motif_problems.load_pdb_motif_problem(motif_start, motif_end, pdb_name=name, base_dir="./")
# out = inpaint_experiment.run_inpainting(
#     exp, target_len, motif_ca_xyz, motif_idcs, exp.diffuser,
#     T=exp.cfg.experiment.T, N_samples_per_diffusion=64, inpainting_task_name=inpainting_task_name,
#     output_dir=sample_dir, inpaint_method="particle", num_save=1)

# %%



