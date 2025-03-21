from sd_pipeline import Decoding_MCTS
from diffusers import DDIMScheduler
import torch
import numpy as np
import random
from PIL import Image
import PIL
from typing import Callable, List, Optional, Union, Dict, Any
from dataset import AVACompressibilityDataset, AVACLIPDataset
from vae import encode
import os
from aesthetic_scorer import AestheticScorerDiff_Time, MLPDiff
import wandb
import argparse
from tqdm import tqdm
import datetime
from compressibility_scorer import CompressibilityScorerDiff, jpeg_compressibility, CompressibilityScorer_modified
from aesthetic_scorer import AestheticScorerDiff
from torch.amp import autocast
import warnings
warnings.filterwarnings("ignore")

def parse():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward", type=str, default='aesthetic')
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--nfe_per_action", type=int, default=2)
    parser.add_argument("--expansion_coef", type=float, default=0.2)
    parser.add_argument("--val_bs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate_size",type=int, default=2)  
    parser.add_argument("--variant", type=str, default="PM")
    parser.add_argument("--valuefunction", type=str, default="")
    parser.add_argument("--progressive_widening", type=bool, default=False)
    parser.add_argument("--pw_alpha", type=float, default=0.9)
    parser.add_argument("--value_gradient", type=bool, default=False)
    parser.add_argument("--kl_lagrangian_coef", type=float, default=0.005)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--tempering_gamma", type=float, default=0.008)
    parser.add_argument("--jump_policy", type=str, default=None)
    args = parser.parse_args()
    return args


######### preparation ##########

args = parse()
if args.progressive_widening == False:
    args.pw_alpha = 0.0
assert not ((args.value_gradient) == True and (args.reward == 'compressibility'))
args.evaluation_budget = args.duplicate_size * args.nfe_per_action

device= args.device
save_file = True

run_name = f"{args.variant}_M={args.duplicate_size}_NFE={args.nfe_per_action}_C={args.expansion_coef}_PW={args.pw_alpha}_G={args.value_gradient}:{args.kl_lagrangian_coef}_J={args.jump_policy}_{args.valuefunction.split('/')[-1] if args.valuefunction != '' else ''}"
unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
run_name = run_name + '_' + unique_id


if args.out_dir == "":
    args.out_dir = 'logs/' + run_name
try:
    os.makedirs(args.out_dir)
except:
    pass


wandb.init(project=f"MCTS-{args.reward}", name=run_name,config=args)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
initial_memory = torch.cuda.memory_allocated()

sd_model = Decoding_MCTS.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
sd_model.to(device)

# switch to DDIM scheduler
sd_model.scheduler = DDIMScheduler.from_config(sd_model.scheduler.config)
sd_model.scheduler.set_timesteps(args.num_inference_steps, device=device)

sd_model.vae.requires_grad_(False)
sd_model.text_encoder.requires_grad_(False)
sd_model.unet.requires_grad_(False)

sd_model.vae.eval()
sd_model.text_encoder.eval()
sd_model.unet.eval()

sd_model.set_progressive_widening(args.progressive_widening)
sd_model.set_pw_alpha(args.pw_alpha)
sd_model.set_value_gradient(args.value_gradient)
sd_model.set_kl_lagrangian_coef(args.kl_lagrangian_coef)
sd_model.set_tempering_gamma(args.tempering_gamma)
sd_model.set_jump_policy(args.jump_policy)

assert args.variant in ['PM', 'MC']

if args.reward == 'compressibility':
    if args.variant == 'PM':
        scorer = CompressibilityScorer_modified(dtype=torch.float32)#.to(device)
    elif args.variant == 'MC':
        scorer = CompressibilityScorerDiff(dtype=torch.float32).to(device)
elif args.reward == 'aesthetic':
    if args.variant == 'PM':
        scorer = AestheticScorerDiff(dtype=torch.float32).to(device)
    elif args.variant == 'MC':
        scorer = AestheticScorerDiff_Time(dtype=torch.float32).to(device)
        if args.valuefunction != "":
            scorer.set_valuefunction(args.valuefunction)
            scorer = scorer.to(device)
else:
    raise ValueError("Invalid reward")

scorer.requires_grad_(False)
scorer.eval()

sd_model.setup_scorer(scorer)
sd_model.set_variant(args.variant)
sd_model.set_reward(args.reward)
sd_model.set_parameters(args.bs, args.duplicate_size)
sd_model.set_nfe_per_action(args.nfe_per_action)
sd_model.set_expansion_coef(args.expansion_coef)


### introducing evaluation prompts
import prompts as prompts_file
eval_prompt_fn = getattr(prompts_file, 'eval_aesthetic_animals')


image = []
eval_prompt_list = []
KL_list = []

for i in tqdm(range(args.num_images // args.bs), desc="Generating Images", position=0):
    wandb.log(
        {"inner_iter": i}
    )
    
    ## Image Seeds
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        shape = (args.num_images//args.bs, args.bs * args.duplicate_size * args.nfe_per_action, 4, 64, 64)
        init_latents = torch.randn(shape, device=device)
    else:
        init_latents = None
    
    if init_latents is None:
        init_i = None
    else:
        init_i = init_latents[i]
    eval_prompts, _ = zip(
        *[eval_prompt_fn() for _ in range(args.bs)]
    )
    eval_prompts = list(eval_prompts)
    eval_prompt_list.extend(eval_prompts)
    with autocast(device_type=device):
        image_, kl_loss = sd_model(eval_prompts, num_images_per_prompt=1, eta=1.0, latents=init_i) # List of PIL.Image objects      
    image.extend(image_)
    KL_list.append(kl_loss)

# KL_entropy = torch.mean(torch.stack(KL_list))

end_event.record()
torch.cuda.synchronize() # Wait for the events to complete
gpu_time = start_event.elapsed_time(end_event)/1000 # Time in seconds
max_memory = torch.cuda.max_memory_allocated()
max_memory_used = (max_memory - initial_memory) / (1024 ** 2)

wandb.log({
        "GPUTimeInS": gpu_time,
        "MaxMemoryInMb": max_memory_used,
    })

###### evaluation and metric #####
if args.reward == 'compressibility':
    gt_dataset= AVACompressibilityDataset(image)
elif args.reward == 'aesthetic':
    from importlib import resources
    ASSETS_PATH = resources.files("assets")
    eval_model = MLPDiff().to(device)
    eval_model.requires_grad_(False)
    eval_model.eval()
    s = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=device, weights_only=True)
    eval_model.load_state_dict(s)
    gt_dataset= AVACLIPDataset(image)    
    
gt_dataloader = torch.utils.data.DataLoader(gt_dataset, batch_size=args.val_bs, shuffle=False)
# # equivalent with the tree.evaluate logic
# import torchvision
# resize = torchvision.transforms.Resize(224)
# normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
# img = normalize(resize(torch.tensor(np.array(image)).permute(0,3,1,2)) / 255).to(device)
# score = scorer(img)[0]
# scorer(normalize(resize(torch.tensor(np.array(image)).permute(0,3,1,2)) / 255).to(device))[0]

# image_path = "/home/jaewoo/research/diffusion-mcts/SVDD-image/001_snail | reward: 8.43678092956543.png"
# IMG = Image.open(image_path).convert("RGB")
# scorer(normalize(resize(torch.tensor(np.array(IMG)[np.newaxis, :, :, :]).permute(0,3,1,2)) / 255).to(device))[0]
# 
# scorer(gt_dataset.processor(images=np.array(IMG)[np.newaxis, :, :, :], return_tensors='pt')['pixel_values'].to(device))[0]

# 데이터셋에서 프로세서 꺼내오기
# gt_dataset.processor(images=gt_dataset.data, return_tensors='pt')['pixel_values'].to(device)
# scorer(gt_dataset.processor(images=gt_dataset.data, return_tensors='pt')['pixel_values'].to(device))[0]
# scorer(gt_dataset.processor(images=np.array(image), return_tensors='pt')['pixel_values'].to(device))[0]

# np.array(gt_dataset.data[0]) == np.array(image[0])

with torch.no_grad():
    eval_rewards = []

    
    if args.reward == 'compressibility':
        jpeg_compressibility_scores = jpeg_compressibility(image)
        scores = torch.tensor(jpeg_compressibility_scores, dtype=image.dtype, device=image.device)
        
    elif args.reward == 'aesthetic':
        import torchvision
        resize = torchvision.transforms.Resize(224, antialias=False)
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        img = normalize(resize(torch.tensor(np.array(image)).permute(0,3,1,2)) / 255).to(device)
        score = scorer(img)[0]
        print(f"eval_{args.reward}_rewards_mean --> tree.evaluate, DAS rewarding!", torch.mean(score))
    
    wandb.log({
        f"das/eval_{args.reward}_rewards_mean": torch.mean(score),
        f"das/eval_{args.reward}_rewards_std": torch.std(score),
        f"das/eval_{args.reward}_rewards_median": torch.median(score),
        f"das/eval_{args.reward}_rewards_90%_quantile": torch.quantile(score, 0.1),
        f"das/eval_{args.reward}_rewards_10%_quantile": torch.quantile(score, 0.9),
    })


with torch.no_grad():
    eval_rewards = []
    
    for inputs in gt_dataloader:
        inputs = inputs.to(device)

        if args.reward == 'compressibility':
            jpeg_compressibility_scores = jpeg_compressibility(inputs)
            scores = torch.tensor(jpeg_compressibility_scores, dtype=inputs.dtype, device=inputs.device)
        
        elif args.reward == 'aesthetic':
            scores = eval_model(inputs)
            scores = scores.squeeze(1)
        
        eval_rewards.extend(scores.tolist())

    eval_rewards = torch.tensor(eval_rewards)


    print(f"eval_{args.reward}_rewards_mean", torch.mean(eval_rewards))

    
    wandb.log({
        f"eval_{args.reward}_rewards_mean": torch.mean(eval_rewards),
        f"eval_{args.reward}_rewards_std": torch.std(eval_rewards),
        f"eval_{args.reward}_rewards_median": torch.median(eval_rewards),
        f"eval_{args.reward}_rewards_90%_quantile": torch.quantile(eval_rewards, 0.1),
        f"eval_{args.reward}_rewards_10%_quantile": torch.quantile(eval_rewards, 0.9),
    })


if save_file:
    images = []
    log_dir = os.path.join(args.out_dir, "eval_vis")
    os.makedirs(log_dir, exist_ok=True)
    np.save(f"{args.out_dir}/scores.npy", eval_rewards)

    # Function to save array to a text file with commas
    def save_array_to_text_file(array, file_path):
        with open(file_path, 'w') as file:
            array_str = ','.join(map(str, array.tolist()))
            file.write(array_str + ',')

    # Save the arrays to text files
    save_array_to_text_file(eval_rewards, f"{args.out_dir}/eval_rewards.txt")
    print("Arrays have been saved to text files.")
    
    for idx, im in enumerate(image):
        prompt = eval_prompt_list[idx]
        reward = eval_rewards[idx]
        
        im.save(f"{log_dir}/{idx:03d}_{prompt}_score={reward:2f}.png")
        
        pil = im.resize((256, 256))

        images.append(wandb.Image(pil, caption=f"{prompt:.25} | score:{reward:.2f}"))

    wandb.log(
        {"images": images}
    )