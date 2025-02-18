import torch, torchvision
import numpy as np
from tqdm import tqdm
from diffusers_patch.ddim_with_kl import predict_x0_from_xt_MCTS, ddim_step_KL_MCTS


class Node:
    def __init__(self, state, reward, timestep, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = 0 ## TODO fix
        self.prior = 0
        self.reward = reward
        self.timestep = timestep 
        self.value = 0
        
    def get_parent(self):
        return self.parent
    
    def get_children(self):
        return self.children
        
    def add_children(self, states, timesteps):
        for state, timestep in zip(states, timesteps):
            self.children.append(Node(state=state, reward=None, timestep=timestep, parent=self))
            
    def set_value(self, value):
        self.value = value
        
    def _terminal_checker(self, max_timestep):
        if self.timestep == max_timestep:
            return True
        else:
            return False
        

class BatchedNode:
    def __init__(self, node_list):
        self.node_list = node_list

    @property
    def batch_size(self):
        return len(self.node_list)

    @property
    def states(self):
        return torch.stack([node.state for node in self.node_list], dim=0)

    @states.setter
    def states(self, new_states):
        assert new_states.shape[0] == self.batch_size, "Batch size mismatch in states setter."
        for i, node in enumerate(self.node_list):
            node.state = new_states[i : i + 1]

    @property
    def timesteps(self):
        # 각 노드의 timestep을 (B, ...) 형태로 결합 (노드마다 shape이 다를 수 있으므로 cat dim은 상황에 맞게 수정)
        return torch.stack([node.timestep for node in self.node_list], dim=0)

    @timesteps.setter
    def timesteps(self, new_timesteps):
        # new_timesteps: (B, ...) 텐서라고 가정
        assert new_timesteps.shape[0] == self.batch_size, "Batch size mismatch in timesteps setter."
        for i, node in enumerate(self.node_list):
            node.timestep = new_timesteps[i : i + 1]
            
    @property
    def rewards(self):
        # 각 노드의 reward를 (B, ...) 형태로 결합합니다.
        return torch.tensor([node.reward for node in self.node_list])

    @rewards.setter
    def rewards(self, new_rewards):
        # new_rewards: (B, ...) 텐서라고 가정합니다.
        assert new_rewards.shape[0] == self.batch_size, "Batch size mismatch in rewards setter."
        for i, node in enumerate(self.node_list):
            node.reward = new_rewards[i : i + 1]
            
    @property
    def values(self):
        return torch.tensor([node.value for node in self.node_list])
    
    @property
    def visit_counts(self):
        return torch.tensor([node.visit_count for node in self.node_list])
            
    def get_children(self):
        return [node.get_children() for node in self.node_list]
    
    def get_novel_children(self):
        result = []
        for node in self.node_list:
            novel_children = []
            children = node.get_children()
            for child in children:
                if child.reward is None:
                    novel_children.append(child)
            result.append(novel_children)
        return result
            
    def add_children(self, children_states_list, children_timesteps_list):
        """
        children_states_list: list of length B, where each element is a list of child state tensors for that node.
        children_timesteps_list: list of length B, where each element is a list of corresponding timesteps.
        """
        # for each node in the batch, add its children via the Node.add_children method.
        # (여기서는 간단하게 zip과 list comprehension을 사용)
        [node.add_children(states, ts) for node, states, ts in zip(self.node_list, children_states_list, children_timesteps_list)]
        
    def __call__(self):
        return self.node_list 
    
    def __getitem__(self, idx):
        return self.node_list[idx]


class TreePolicy:
    def __init__(
            self, 
            initial_children, 
            select_function, 
            pipeline, 
            do_classifier_free_guidance,
            prompt_embeds=None, 
            cross_attention_kwargs=None,
            guidance_scale=1.0,
            eta=1.0,
            expansion_coef=0.2
        ):
        self.prompt_embeds = prompt_embeds
        self.cross_attention_kwargs = cross_attention_kwargs
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.select_function = select_function
        self.progressive_widening = False
        self.pipeline = pipeline
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.expansion_coef = expansion_coef 
        self.exploration_constant = expansion_coef  # UCT 상수로 사용
        self.max_timestep = self.pipeline.scheduler.timesteps[-1] 
        # initial_children: torch.Tensor of shape (B * duplicate, C, H, W)
        node_list = [Node(state=None, timestep=None, parent=None, reward=None) for _ in range(pipeline.batch_size)]
        self.device = pipeline.device
        
        # initial node for x_T starting point
        self.root_nodes = BatchedNode(node_list)
        self.root_nodes.add_children(
            initial_children.view(pipeline.batch_size, pipeline.duplicate, *initial_children.shape[1:]), 
            torch.ones(pipeline.batch_size, pipeline.duplicate, device=self.device) * pipeline.scheduler.timesteps[0]
        )
        
        for nodes in tqdm(list(zip(*self.root_nodes.get_novel_children())), desc='Initial Evaluating', leave=False, position=3):
            self.evaluate(BatchedNode(nodes))
        self.backpropagate(self.root_nodes.get_children())
        


    def select(self):
        """
        각 트리의 root부터 시작하여, 각 트리마다 leaf(또는 terminal) 노드까지 UCT 선택을 진행합니다.
        어떤 트리는 다른 트리보다 일찍 선택 과정이 끝날 수 있으므로, 각 트리에 대해 개별적으로 처리한 후
        결과를 BatchedNode 객체로 반환합니다.
        """
        selected_nodes = []
        # self.root_nodes.node_list는 배치 내 개별 root Node들의 리스트입니다.
        for node in self.root_nodes:
            current = node
            # 자식이 존재하고 현재 노드가 leaf(terminal이 아님) 상태이면 계속 진행
            while current.get_children() and not current._terminal_checker(self.max_timestep):
                # 부모의 방문 횟수 (0이면 1로 대체)
                parent_visits = current.visit_count if current.visit_count > 0 else 1
                parent_visits_tensor = torch.tensor(parent_visits, dtype=torch.float32, device=self.device)
                # 자식들의 value와 visit_count를 torch 텐서로 만듭니다.
                child_values = torch.tensor([child.value for child in current.get_children()],
                                            dtype=torch.float32, device=self.device)
                child_visits = torch.tensor([child.visit_count for child in current.get_children()],
                                            dtype=torch.float32, device=self.device)
                # UCT 값 계산: avg_value + exploration term
                uct_values = child_values / child_visits + \
                            self.exploration_constant * torch.sqrt(torch.log(parent_visits_tensor) / child_visits)
                # 가장 높은 UCT 값을 가진 자식의 인덱스를 얻습니다.
                best_idx = torch.argmax(uct_values).item()
                best_child = current.get_children()[best_idx]
                current = best_child
            selected_nodes.append(current)
        # 선택된 노드들을 BatchedNode로 묶어 반환합니다.
        return BatchedNode(selected_nodes)
            
    
    def expand(self, nodes):
        """
        확장 단계: BatchedNode인 nodes의 각 노드(state)를 바탕으로 자식 노드들을 생성합니다.
        여기서는 각 노드가 pipeline.duplicate 만큼의 자식을 갖는다고 가정합니다.
        t와 latents는 현재 timestep과 latent 정보를 의미합니다.
        """
        # nodes.states: (B, C, H, W)
        if self.do_classifier_free_guidance:
            latent_model_input = torch.cat([nodes.states] * 2, dim=0)  # (2B, C, H, W)
        else:
            latent_model_input = nodes.states  # (B, C, H, W)
        
        # scale_model_input는 배치 지원
        latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, nodes.timesteps)
        noise_pred = self.pipeline.unet(
            latent_model_input,
            nodes.timesteps.repeat_interleave(2) if self.do_classifier_free_guidance else nodes.timesteps,
            encoder_hidden_states=self.prompt_embeds,
            cross_attention_kwargs=self.cross_attention_kwargs,
        ).sample

        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            old_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        noise_pred = old_noise_pred 
            
        # ddim_step_KL_modified: 노드의 상태에서 새로운 latent 후보들을 생성
        # new_latents: (B * duplicate, C, H, W)
        duplicate = self.pipeline.duplicate
        new_latents, kl_terms = ddim_step_KL_MCTS( ## TODO 입력 noise_pred 확인
            self.pipeline.scheduler,
            noise_pred,    # 예측된 노이즈
            old_noise_pred,
            nodes.timesteps,
            nodes.states,
            eta=self.eta,
            duplicate=duplicate
        ) # (B * duplicate, C, H, W)
        new_latents = new_latents.view(self.pipeline.batch_size, duplicate, *new_latents.shape[1:])

        step_offset = self.pipeline.scheduler.config.num_train_timesteps // self.pipeline.scheduler.num_inference_steps
        new_timesteps = nodes.timesteps - step_offset
        mask = new_timesteps >= 0
        new_timesteps = torch.where(
            mask, 
            new_timesteps, 
            torch.zeros_like(new_timesteps, device=new_timesteps.device) 
        ).repeat_interleave(duplicate).view(self.pipeline.batch_size, duplicate)
        
        # 각 배치별로 자식 노드 리스트 생성: 각 부모는 duplicate 개의 자식을 가짐.
        nodes.add_children(new_latents, new_timesteps)
        # 평가(evaluate) 단계: 각 부모의 자식들에 대해 reward 계산 
        for nodes in tqdm(list(zip(*nodes.get_novel_children())), desc='Evaluating', leave=False, position=3):
            self.evaluate(BatchedNode(nodes))


    @torch.no_grad()
    def evaluate(self, batched_nodes):
        if  self.pipeline.variant == 'PM' and self.pipeline.reward == 'compressibility':
            latent_model_input = torch.cat([batched_nodes.states] * 2) 
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, batched_nodes.timesteps)
            noise_pred = self.pipeline.unet(
                latent_model_input, 
                batched_nodes.timesteps.repeat_interleave(2) if self.do_classifier_free_guidance else batched_nodes.timesteps, 
                encoder_hidden_states= self.prompt_embeds, 
                cross_attention_kwargs=self.cross_attention_kwargs
            ).sample                       
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            new_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond) 
            
            pred_original_sample = predict_x0_from_xt_MCTS(
                                self.pipeline.scheduler,
                                new_noise_pred,   
                                batched_nodes.timesteps,
                                batched_nodes.states
            )
            
            images = self.pipeline.decode_latents(pred_original_sample)
            images = (images * 255).round().astype("uint8")
            evaluation = self.pipeline.scorer(images)
            batched_nodes.rewards = evaluation
            return evaluation     

        ## Calculate E[x_0|x_t]
        if self.pipeline.variant == 'PM':
            latent_model_input = torch.cat([batched_nodes.states] * 2) 
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, batched_nodes.timesteps)
            noise_pred = self.pipeline.unet(
                latent_model_input, 
                batched_nodes.timesteps.repeat_interleave(2) if self.do_classifier_free_guidance else batched_nodes.timesteps, 
                encoder_hidden_states= self.prompt_embeds, 
                cross_attention_kwargs=self.cross_attention_kwargs
            ).sample             
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            new_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond) 
            pred_original_sample = predict_x0_from_xt_MCTS(
                                self.pipeline.scheduler,
                                new_noise_pred,   
                                batched_nodes.timesteps,
                                batched_nodes.states
            )
            im_pix_un = self.pipeline.vae.decode(pred_original_sample.to(self.pipeline.vae.dtype) / 0.18215).sample

        elif self.pipeline.variant == 'MC':
            im_pix_un = self.pipeline.vae.decode(batched_nodes.states.to(self.pipeline.vae.dtype) / 0.18215).sample
            
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 

        # resize = torchvision.transforms.Resize(224, antialias=False)
        if self.pipeline.reward == 'compressibility':
            resize = torchvision.transforms.Resize(512, antialias=False)
        elif self.pipeline.reward == 'aesthetic':
            resize = torchvision.transforms.Resize(224, antialias=False)
        else:
            raise ValueError('Invalid reward type')
            
        im_pix = resize(im_pix)
        normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        
        if self.pipeline.variant == 'PM':
            evaluation, _ = self.pipeline.scorer(im_pix)
        elif self.pipeline.variant == 'MC':
            if self.pipeline.reward == 'compressibility':
                evaluation, _ = self.pipeline.scorer(batched_nodes.states, timesteps=batched_nodes.timesteps) ### TODO timestep!!! 
            else:
                evaluation, _ = self.pipeline.scorer(im_pix, timesteps=batched_nodes.timesteps)
        batched_nodes.rewards = evaluation
        return evaluation
        
    def backpropagate(self,  batched_children):
        for children in batched_children:
            for child in children:
                r = child.reward  
                current = child
                while current is not None:
                    current.visit_count += 1
                    current.value += r
                    current = current.get_parent()

    def _free_subtree(self, node):
        for child in node.get_children():
            self._free_subtree(child)
        if hasattr(node, 'state') and isinstance(node.state, torch.Tensor):
            if node.state.is_cuda:
                node.state.detach()
            node.state = None
        node.children.clear()
        node.parent = None

    def act_and_prune(self, prune=True):
        selected_nodes = []
        for node in self.root_nodes:
            current = node
            children = current.get_children()
            if children: 
                # 부모의 visit_count가 0이면 1로 대체
                parent_visits = current.visit_count if current.visit_count > 0 else 1
                parent_visits_tensor = torch.tensor(parent_visits, dtype=torch.float32, device=self.device)
                
                # 각 자식의 value와 visit_count를 torch 텐서로 변환
                child_values = torch.tensor([child.value for child in children],
                                            dtype=torch.float32, device=self.device)
                child_visits = torch.tensor([child.visit_count if child.visit_count > 0 else 1 for child in children],
                                            dtype=torch.float32, device=self.device)

                uct_values = child_values / child_visits # + self.exploration_constant * torch.sqrt(torch.log(parent_visits_tensor) / child_visits)
                best_idx = torch.argmax(uct_values).item()
                best_child = children[best_idx]
                # 선택되지 않은 자식들은 재귀적으로 메모리 해제
                if prune:
                    for i, child in enumerate(children):
                        if i != best_idx:
                            self._free_subtree(child)
                current = best_child
            selected_nodes.append(current)
        # 새로운 루트 노드들로 업데이트
        self.root_nodes = BatchedNode(selected_nodes)
        # GPU 메모리 비우기
        if torch.cuda.is_available():
            torch.cuda.empty_cache()