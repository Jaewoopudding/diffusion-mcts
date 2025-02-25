CUDA_VISIBLE_DEVICES=0 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 5 --nfe_per_action 16 --num_images 36 --expansion_coef 0.2 --variant PM &
CUDA_VISIBLE_DEVICES=1 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 8 --nfe_per_action 10 --num_images 36 --expansion_coef 0.2 --variant PM &
CUDA_VISIBLE_DEVICES=2 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 10 --nfe_per_action 8 --num_images 36 --expansion_coef 0.2 --variant PM &
CUDA_VISIBLE_DEVICES=3 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 20 --nfe_per_action 4 --num_images 36 --expansion_coef 0.2 --variant PM &
CUDA_VISIBLE_DEVICES=4 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 40 --nfe_per_action 2 --num_images 36 --expansion_coef 0.2 --variant PM &
CUDA_VISIBLE_DEVICES=5 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 80 --nfe_per_action 1 --num_images 36 --expansion_coef 0.0 --variant PM &
wait

CUDA_VISIBLE_DEVICES=0 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 5 --nfe_per_action 16 --num_images 36 --expansion_coef 0.1 --variant PM &
CUDA_VISIBLE_DEVICES=1 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 8 --nfe_per_action 10 --num_images 36 --expansion_coef 0.1 --variant PM &
CUDA_VISIBLE_DEVICES=2 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 10 --nfe_per_action 8 --num_images 36 --expansion_coef 0.1 --variant PM &
CUDA_VISIBLE_DEVICES=3 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 20 --nfe_per_action 4 --num_images 36 --expansion_coef 0.1 --variant PM &
CUDA_VISIBLE_DEVICES=4 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 40 --nfe_per_action 2 --num_images 36 --expansion_coef 0.1 --variant PM &
CUDA_VISIBLE_DEVICES=5 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 60 --nfe_per_action 1 --num_images 36 --expansion_coef 0.0 --variant PM &
wait

CUDA_VISIBLE_DEVICES=0 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 5 --nfe_per_action 16 --num_images 36 --expansion_coef 0.01 --variant PM &
CUDA_VISIBLE_DEVICES=1 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 8 --nfe_per_action 10 --num_images 36 --expansion_coef 0.01 --variant PM &
CUDA_VISIBLE_DEVICES=2 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 10 --nfe_per_action 8 --num_images 36 --expansion_coef 0.01 --variant PM &
CUDA_VISIBLE_DEVICES=3 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 20 --nfe_per_action 4 --num_images 36 --expansion_coef 0.01 --variant PM &
CUDA_VISIBLE_DEVICES=4 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 40 --nfe_per_action 2 --num_images 36 --expansion_coef 0.01 --variant PM &
CUDA_VISIBLE_DEVICES=5 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 20 --nfe_per_action 1 --num_images 36 --expansion_coef 0.0 --variant PM &
wait

CUDA_VISIBLE_DEVICES=0 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 5 --nfe_per_action 16 --num_images 36 --expansion_coef 0.3 --variant PM &
CUDA_VISIBLE_DEVICES=1 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 8 --nfe_per_action 10 --num_images 36 --expansion_coef 0.3 --variant PM &
CUDA_VISIBLE_DEVICES=2 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 10 --nfe_per_action 8 --num_images 36 --expansion_coef 0.3 --variant PM &
CUDA_VISIBLE_DEVICES=3 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 20 --nfe_per_action 4 --num_images 36 --expansion_coef 0.3 --variant PM &
CUDA_VISIBLE_DEVICES=4 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 40 --nfe_per_action 2 --num_images 36 --expansion_coef 0.3 --variant PM &
CUDA_VISIBLE_DEVICES=5 python inference_decoding_mcts.py --reward 'compressibility' --bs 4 --duplicate_size 40 --nfe_per_action 1 --num_images 36 --expansion_coef 0.0 --variant PM &
wait


CUDA_VISIBLE_DEVICES=6 python inference_decoding_mcts.py --reward 'compressibility' --bs 2 --duplicate_size 2 --nfe_per_action 1 --num_images 2 --expansion_coef 0.0 --variant PM 
