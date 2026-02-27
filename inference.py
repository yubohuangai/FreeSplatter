"""
Lean inference script for FreeSplatter.
Only loads the models you actually need, avoiding the Gradio overhead and OOM issues.

Usage:
  # Image-to-3D using Zero123++ v1.2 (default)
  python inference.py --input image.png --output_dir output/

  # Image-to-3D using Hunyuan3D
  python inference.py --input image.png --output_dir output/ --model hunyuan3d

  # Multi-view to 3D (provide multiple views)
  python inference.py --input view1.png view2.png view3.png --output_dir output/ --mode views_to_3d

  # Two-view scene reconstruction
  python inference.py --input img1.png img2.png --output_dir output/ --mode scene

  # Use specific GPU
  CUDA_VISIBLE_DEVICES=1 python inference.py --input image.png --output_dir output/
"""

import os
import argparse
import time
import numpy as np
import torch
import trimesh
import fpsample
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange
from scipy.spatial.transform import Rotation
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from torchvision.transforms import v2
from pytorch_lightning import seed_everything

from freesplatter.utils.mesh_optim import optimize_mesh
from freesplatter.utils.camera_util import *
from freesplatter.utils.recon_util import *
from freesplatter.utils.infer_util import *


def inv_sigmoid(x):
    return torch.log(x / (1.0 - x))


def save_gaussian(latent, gs_vis_path, model, opacity_threshold=None, pad_2dgs_scale=True):
    if latent.ndim == 3:
        latent = latent[0]
    sh_dim = model.sh_dim
    scale_dim = 2 if model.use_2dgs else 3
    xyz, features, opacity, scaling, rotation = latent.split([3, sh_dim, 1, scale_dim, 4], dim=-1)
    features = features.reshape(features.shape[0], sh_dim // 3, 3)

    if opacity_threshold is not None:
        index = torch.nonzero(opacity.sigmoid() > opacity_threshold)[:, 0]
        xyz, features, opacity, scaling, rotation = xyz[index], features[index], opacity[index], scaling[index], rotation[index]

    cam2world = create_camera_to_world(torch.tensor([0, -2, 0]), camera_system='opencv').to(latent)
    R, T = cam2world[:3, :3], cam2world[:3, 3].reshape(1, 3)
    xyz = xyz @ R.T + T
    rot_np = rotation.detach().cpu().numpy()
    rot_np = Rotation.from_quat(rot_np[:, [1, 2, 3, 0]]).as_matrix()
    rot_np = R.detach().cpu().numpy() @ rot_np
    rotation = torch.from_numpy(Rotation.from_matrix(rot_np).as_quat()[:, [3, 0, 1, 2]]).to(latent)

    if scaling.shape[-1] == 2 and pad_2dgs_scale:
        z_scaling = inv_sigmoid(torch.ones_like(scaling[:, :1]) * 0.001)
        scaling = torch.cat([scaling, z_scaling], dim=-1)

    pc_vis = model.gs_renderer.gaussian_model.set_data(
        xyz.float(), features.float(), scaling.float(), rotation.float(), opacity.float())
    pc_vis.save_ply_vis(gs_vis_path)


def load_freesplatter_model(config_file, ckpt_filename, device):
    ckpt_path = hf_hub_download('TencentARC/FreeSplatter', repo_type='model',
                                 filename=ckpt_filename, local_dir='./ckpts/FreeSplatter')
    model = instantiate_from_config(OmegaConf.load(config_file).model)
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def load_rembg(device):
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True, cache_dir='ckpts/')
    return model.eval().to(device)


def load_diffusion_model(model_name, device):
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    if model_name == 'zero123pp_v11':
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1",
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16, cache_dir="ckpts/")
    elif model_name == 'zero123pp_v12':
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16, cache_dir="ckpts/")
    elif model_name == 'hunyuan3d':
        from freesplatter.hunyuan.hunyuan3d_mvd_std_pipeline import HunYuan3D_MVD_Std_Pipeline
        os.makedirs('./ckpts/Hunyuan3D-1', exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download('tencent/Hunyuan3D-1', repo_type='model', local_dir='./ckpts/Hunyuan3D-1')
        pipeline = HunYuan3D_MVD_Std_Pipeline.from_pretrained(
            './ckpts/Hunyuan3D-1/mvd_std', torch_dtype=torch.float16, use_safetensors=True)
        return pipeline.to(device)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing')
    return pipeline.to(device)


def generate_multiview(pipeline, input_image, model_name, diffusion_steps=30, guidance_scale=4.0, seed=42):
    seed_everything(seed)
    if model_name == 'hunyuan3d':
        output_image = pipeline(input_image, num_inference_steps=diffusion_steps,
                                guidance_scale=guidance_scale, guidance_curve=lambda t: 2.0).images[0]
    else:
        output_image = pipeline(input_image, num_inference_steps=diffusion_steps,
                                guidance_scale=guidance_scale).images[0]
    return output_image


def reconstruct_object(freesplatter, images, alphas, device, output_dir, mesh_reduction=0.5):
    images, alphas = images.to(device), alphas.to(device)

    t0 = time.time()
    with torch.inference_mode():
        gaussians = freesplatter.forward_gaussians(images.unsqueeze(0))
    t1 = time.time()

    c2ws_pred, focals_pred = freesplatter.estimate_poses(
        images, gaussians, masks=alphas, use_first_focal=True, pnp_iter=10)
    t2 = time.time()

    gs_vis_path = os.path.join(output_dir, 'gs_vis.ply')
    save_gaussian(gaussians, gs_vis_path, freesplatter, opacity_threshold=5e-3, pad_2dgs_scale=True)
    print(f'Saved gaussian at {gs_vis_path}')

    with torch.inference_mode():
        c2ws_video = get_circular_cameras(N=120, elevation=0, radius=2.0, normalize=True).to(device)
        fx = fy = focals_pred.mean() / 512.0
        cx = cy = torch.ones_like(fx) * 0.5
        fxfycxcy_video = torch.tensor([fx, fy, cx, cy]).unsqueeze(0).repeat(c2ws_video.shape[0], 1).to(device)
        video_frames = freesplatter.forward_renderer(
            gaussians, c2ws_video.unsqueeze(0), fxfycxcy_video.unsqueeze(0))['image'][0].clamp(0, 1)

    video_path = os.path.join(output_dir, 'gs.mp4')
    save_video(video_frames, video_path, fps=30)
    print(f'Saved video at {video_path}')
    t3 = time.time()

    with torch.inference_mode():
        c2ws_fusion = get_fibonacci_cameras(N=120, radius=2.0)
        c2ws_fusion, _ = normalize_cameras(c2ws_fusion, camera_position=torch.tensor([0., -2., 0.]), camera_system='opencv')
        c2ws_fusion = c2ws_fusion.to(device)
        c2ws_fusion_reference = torch.linalg.inv(c2ws_fusion[0:1]) @ c2ws_fusion
        fx = fy = focals_pred.mean() / 512.0
        cx = cy = torch.ones_like(fx) * 0.5
        fov = np.rad2deg(np.arctan(0.5 / fx.item())) * 2
        fxfycxcy_fusion = torch.tensor([fx, fy, cx, cy]).unsqueeze(0).repeat(c2ws_fusion.shape[0], 1).to(device)

        fusion_render_results = freesplatter.forward_renderer(
            gaussians, c2ws_fusion_reference.unsqueeze(0), fxfycxcy_fusion.unsqueeze(0))
        images_fusion = fusion_render_results['image'][0].clamp(0, 1).permute(0, 2, 3, 1)
        alphas_fusion = fusion_render_results['alpha'][0].permute(0, 2, 3, 1)
        depths_fusion = fusion_render_results['depth'][0].permute(0, 2, 3, 1)

        fusion_images = (images_fusion.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        fusion_depths = depths_fusion.detach().cpu().numpy()
        fusion_alphas = alphas_fusion.detach().cpu().numpy()
        fusion_masks = (fusion_alphas > 1e-2).astype(np.uint8)
        fusion_depths = fusion_depths * fusion_masks - np.ones_like(fusion_depths) * (1 - fusion_masks)
        fusion_c2ws = c2ws_fusion.detach().cpu().numpy()

        mesh_path = os.path.join(output_dir, 'mesh.obj')
        rgbd_to_mesh(fusion_images, fusion_depths, fusion_c2ws, fov, mesh_path, cam_elev_thr=-90)
        print(f'Saved mesh at {mesh_path}')
    t4 = time.time()

    cam_pos = c2ws_fusion[:, :3, 3].cpu().numpy()
    cam_inds = torch.from_numpy(fpsample.fps_sampling(cam_pos, 16).astype(int)).to(device=device)
    alphas_bake = alphas_fusion[cam_inds]
    images_bake = (images_fusion[cam_inds] - (1 - alphas_bake)) / alphas_bake.clamp(min=1e-6)
    fxfycxcy = fxfycxcy_fusion[cam_inds].clone()
    intrinsics = torch.eye(3).unsqueeze(0).repeat(len(cam_inds), 1, 1).to(fxfycxcy)
    intrinsics[:, 0, 0] = fxfycxcy[:, 0]
    intrinsics[:, 0, 2] = fxfycxcy[:, 2]
    intrinsics[:, 1, 1] = fxfycxcy[:, 1]
    intrinsics[:, 1, 2] = fxfycxcy[:, 3]

    out_mesh = trimesh.load(str(mesh_path), process=False)
    with torch.enable_grad():
        out_mesh = optimize_mesh(
            out_mesh, images_bake, alphas_bake.squeeze(-1),
            c2ws_fusion[cam_inds].inverse(), intrinsics,
            simplify=mesh_reduction, verbose=False)
    mesh_fine_path = os.path.join(output_dir, 'mesh.glb')
    out_mesh.export(mesh_fine_path)
    print(f'Saved optimized mesh at {mesh_fine_path}')
    t5 = time.time()

    print(f'\nTiming:')
    print(f'  Generate Gaussians: {t1 - t0:.2f}s')
    print(f'  Estimate poses:     {t2 - t1:.2f}s')
    print(f'  Render video:       {t3 - t2:.2f}s')
    print(f'  Extract mesh:       {t4 - t3:.2f}s')
    print(f'  Optimize mesh:      {t5 - t4:.2f}s')
    print(f'  Total:              {t5 - t0:.2f}s')

    return gs_vis_path, video_path, mesh_fine_path


def reconstruct_scene(freesplatter, images, device, output_dir):
    images = images.to(device)

    t0 = time.time()
    with torch.inference_mode():
        gaussians = freesplatter.forward_gaussians(images.unsqueeze(0))
    t1 = time.time()

    c2ws_pred, focals_pred = freesplatter.estimate_poses(images, gaussians, use_first_focal=True, pnp_iter=10)
    baseline_pred = (c2ws_pred[:, :3, 3] - c2ws_pred[:1, :3, 3]).norm() + 1e-2
    scale_factor = 1.0 / baseline_pred
    c2ws_pred = c2ws_pred.clone()
    c2ws_pred[:, :3, 3] *= scale_factor
    t2 = time.time()

    gs_vis_path = os.path.join(output_dir, 'gs_vis.ply')
    save_gaussian(gaussians, gs_vis_path, freesplatter, opacity_threshold=5e-3)
    print(f'Saved gaussian at {gs_vis_path}')

    with torch.inference_mode():
        c2ws_video = generate_interpolated_path(c2ws_pred.detach().cpu().numpy()[:, :3, :], n_interp=120)
        c2ws_video = torch.cat([
            torch.from_numpy(c2ws_video),
            torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(c2ws_video.shape[0], 1, 1)
        ], dim=1).to(gaussians)
        fx = fy = focals_pred.mean() / 512.0
        cx = cy = torch.ones_like(fx) * 0.5
        fxfycxcy_video = torch.tensor([fx, fy, cx, cy]).unsqueeze(0).repeat(c2ws_video.shape[0], 1).to(device)
        video_frames = freesplatter.forward_renderer(
            gaussians, c2ws_video.unsqueeze(0), fxfycxcy_video.unsqueeze(0),
            rescale=scale_factor.reshape(1).to(gaussians))['image'][0].clamp(0, 1)

    video_path = os.path.join(output_dir, 'gs.mp4')
    save_video(video_frames, video_path, fps=30)
    print(f'Saved video at {video_path}')
    t3 = time.time()

    print(f'\nTiming:')
    print(f'  Generate Gaussians: {t1 - t0:.2f}s')
    print(f'  Estimate poses:     {t2 - t1:.2f}s')
    print(f'  Render video:       {t3 - t2:.2f}s')
    print(f'  Total:              {t3 - t0:.2f}s')

    return gs_vis_path, video_path


def main():
    parser = argparse.ArgumentParser(description='FreeSplatter Inference')
    parser.add_argument('--input', nargs='+', required=True, help='Input image(s)')
    parser.add_argument('--output_dir', type=str, default='output/', help='Output directory')
    parser.add_argument('--mode', type=str, default='img_to_3d',
                        choices=['img_to_3d', 'views_to_3d', 'scene'],
                        help='img_to_3d: single image to 3D, views_to_3d: multi-view to 3D, scene: two-view scene')
    parser.add_argument('--model', type=str, default='zero123pp_v12',
                        choices=['zero123pp_v11', 'zero123pp_v12', 'hunyuan3d'],
                        help='Diffusion model for multi-view generation (only for img_to_3d mode)')
    parser.add_argument('--gs_type', type=str, default='2DGS', choices=['2DGS', '3DGS'])
    parser.add_argument('--mesh_reduction', type=float, default=0.5)
    parser.add_argument('--diffusion_steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_rembg', action='store_true', help='Skip background removal')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda')
    torch.set_grad_enabled(False)

    if args.mode == 'img_to_3d':
        print(f'=== Image-to-3D with {args.model} + FreeSplatter ({args.gs_type}) ===')
        input_image = Image.open(args.input[0])

        # Step 1: Remove background
        if not args.no_rembg:
            print('Loading background remover...')
            rembg_model = load_rembg(device)
            input_image = remove_background(input_image, rembg_model)
            del rembg_model
            torch.cuda.empty_cache()
            print('Background removed.')

        # Step 2: Generate multi-view images
        print(f'Loading diffusion model ({args.model})...')
        diffusion = load_diffusion_model(args.model, device)
        input_image_resized = resize_foreground(input_image, 0.9)
        print('Generating multi-view images...')
        output_image = generate_multiview(
            diffusion, input_image_resized, args.model,
            args.diffusion_steps, args.guidance_scale, args.seed)

        # Free diffusion model before loading freesplatter
        del diffusion
        torch.cuda.empty_cache()

        # Step 3: Preprocess views
        print('Loading background remover for view processing...')
        rembg_model = load_rembg(device)
        image, alpha = rgba_to_white_background(input_image_resized)
        image = v2.functional.resize(image, 512, interpolation=3, antialias=True).clamp(0, 1)
        alpha = v2.functional.resize(alpha, 512, interpolation=0, antialias=True).clamp(0, 1)

        output_image_rgba = remove_background(output_image, rembg_model)
        del rembg_model
        torch.cuda.empty_cache()

        is_hunyuan = (args.model == 'hunyuan3d')
        if not is_hunyuan:
            images, alphas = rgba_to_white_background(output_image_rgba)
        else:
            _, alphas = rgba_to_white_background(output_image_rgba)
            images = torch.from_numpy(np.asarray(output_image) / 255.0).float()
            images = rearrange(images, 'h w c -> c h w')

        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        alphas = rearrange(alphas, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        if is_hunyuan:
            images = images[[0, 2, 4, 5, 3, 1]]
            alphas = alphas[[0, 2, 4, 5, 3, 1]]
        images = v2.functional.resize(images, 512, interpolation=3, antialias=True).clamp(0, 1)
        alphas = v2.functional.resize(alphas, 512, interpolation=0, antialias=True).clamp(0, 1)

        images = torch.cat([image.unsqueeze(0), images], dim=0)
        alphas = torch.cat([alpha.unsqueeze(0), alphas], dim=0)
        view_indices = [1, 2, 3, 4, 5, 6]
        images, alphas = images[view_indices], alphas[view_indices]

        # Step 4: Reconstruct 3D
        gs_type = args.gs_type
        config_file = 'configs/freesplatter-object-2dgs.yaml' if gs_type == '2DGS' else 'configs/freesplatter-object.yaml'
        ckpt_file = 'freesplatter-object-2dgs.safetensors' if gs_type == '2DGS' else 'freesplatter-object.safetensors'
        print(f'Loading FreeSplatter ({gs_type})...')
        freesplatter = load_freesplatter_model(config_file, ckpt_file, device)

        print('Reconstructing 3D...')
        reconstruct_object(freesplatter, images, alphas, device, args.output_dir, args.mesh_reduction)

    elif args.mode == 'views_to_3d':
        print(f'=== Multi-view to 3D with FreeSplatter ({args.gs_type}) ===')

        if not args.no_rembg:
            print('Loading background remover...')
            rembg_model = load_rembg(device)

        images_list, alphas_list = [], []
        for img_path in args.input:
            img = Image.open(img_path)
            w, h = img.size
            if not args.no_rembg:
                img_rgba = remove_background(img, rembg_model)
            else:
                img_rgba = img
            if img.mode == 'RGBA':
                im, al = rgba_to_white_background(img_rgba)
                im = v2.functional.center_crop(im, min(h, w))
                al = v2.functional.center_crop(al, min(h, w))
            else:
                img_rgba = resize_foreground(img_rgba, 0.9)
                im, al = rgba_to_white_background(img_rgba)
            im = v2.functional.resize(im, 512, interpolation=3, antialias=True).clamp(0, 1)
            al = v2.functional.resize(al, 512, interpolation=0, antialias=True).clamp(0, 1)
            images_list.append(im)
            alphas_list.append(al)

        if not args.no_rembg:
            del rembg_model
            torch.cuda.empty_cache()

        images = torch.stack(images_list, dim=0)
        alphas = torch.stack(alphas_list, dim=0)

        gs_type = args.gs_type
        config_file = 'configs/freesplatter-object-2dgs.yaml' if gs_type == '2DGS' else 'configs/freesplatter-object.yaml'
        ckpt_file = 'freesplatter-object-2dgs.safetensors' if gs_type == '2DGS' else 'freesplatter-object.safetensors'
        print(f'Loading FreeSplatter ({gs_type})...')
        freesplatter = load_freesplatter_model(config_file, ckpt_file, device)

        print('Reconstructing 3D...')
        reconstruct_object(freesplatter, images, alphas, device, args.output_dir, args.mesh_reduction)

    elif args.mode == 'scene':
        print(f'=== Two-view Scene Reconstruction ===')
        assert len(args.input) == 2, 'Scene mode requires exactly 2 input images'

        images_list = []
        for img_path in args.input:
            img = Image.open(img_path)
            w, h = img.size
            img_t = torch.from_numpy(np.asarray(img) / 255.0).float()
            img_t = rearrange(img_t, 'h w c -> c h w')
            img_t = v2.functional.center_crop(img_t, min(h, w))
            img_t = v2.functional.resize(img_t, 512, interpolation=3, antialias=True).clamp(0, 1)
            images_list.append(img_t)

        images = torch.stack(images_list, dim=0)

        print('Loading FreeSplatter-Scene...')
        freesplatter = load_freesplatter_model(
            'configs/freesplatter-scene.yaml', 'freesplatter-scene.safetensors', device)

        print('Reconstructing scene...')
        reconstruct_scene(freesplatter, images, device, args.output_dir)

    print(f'\nDone! Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
