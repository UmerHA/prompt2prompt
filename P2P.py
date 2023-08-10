import abc
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
import numpy as np
import torch

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput


class LoggedVars:
    def __init__(self): self.variables = {}
    def __setitem__(self,k,v): self.variables[k] =  v.detach().to('cpu').clone() if hasattr(v,'detach') else v
    def __getitem__(self, k): return self.variables[k] 
    def __getattr__(self,k): return self.variables[k]
    def keys(self): return self.variables.keys()
    def items(self): return self.variables.items()
    def clear(self): return self.variables.clear()

class AttentionControl(abc.ABC):
    def step_callback(self, x_t): return x_t
    def between_steps(self): return
    @property
    def num_uncond_att_layers(self): return 0
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str): raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]   
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet) 
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

def view_images(images, num_rows=1, offset_ratio=0.02, display_image=True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list: num_empty = len(images) % num_rows
    elif images.ndim == 4: num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image: display(pil_img)
    return pil_img


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store(): return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2: self.step_store[key].append(attn) # avoid memory overhead
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0: self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])): self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self): return {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
    
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None: x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2: return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else: return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace): raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        # FIXME not replace correctly
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else: attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend,
                 tokenizer,
                 device):
        super(AttentionControlEdit, self).__init__()
        # add tokenizer and device here

        self.tokenizer = tokenizer
        self.device = device

        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, self.tokenizer).to(self.device)
        if type(self_replace_steps) is float: self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend  # 在外面定义后传进来

class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend = None, tokenizer=None, device=None, logger:LoggedVars=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device)
        self.mapper = get_replacement_mapper(prompts, self.tokenizer).to(self.device)
        self.logger = logger
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if self.logger is not None:
            self.logger.clear()
            self.logger['mapper'] = self.mapper
            self.logger['self'] = self
            self.logger['attn'] = attn
            self.logger['is_cross'] = is_cross
            self.logger['place_in_unet'] = place_in_unet
        return super(AttentionReplace, self).forward(attn, is_cross, place_in_unet)


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict: cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps: cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0: alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float: bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None: word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    words_x,words_y = x.split(' '), y.split(' ')
    if len(words_x) != len(words_y): raise ValueError(f"Prompts need same lengths, but are {len(words_x)} and {len(words_y)} words.")
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_: mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1
    return torch.from_numpy(mapper).float()

def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)

def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str: word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int: word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place: out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


class Prompt2PromptPipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]

    @classmethod
    def from_pretrained(cls, *args, attn_logger:LoggedVars=None, **kwargs):
        pipe = super().from_pretrained(*args, **kwargs)
        pipe.attn_logger = attn_logger
        return pipe
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        controller: AttentionStore = None,  # todo: don't pass in controller, but use cross_attention_kwargs
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        self.register_attention_control(controller) # add attention controller
        height,width = (height or self.unet.config.sample_size * self.vae_scale_factor, width or self.unet.config.sample_size * self.vae_scale_factor) # 0. Default height and width to unet
        self.check_inputs(prompt, height, width, callback_steps) # 1. Check inputs. Raise error if not correct

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # following lines are missing:
        # text_encoder_lora_scale = (
        #   cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        # )
        text_embeddings = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents( batch_size * num_images_per_prompt, num_channels_latents, height, width, text_embeddings.dtype, device, generator, latents,)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta) # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample # predict the noise residual
                if do_classifier_free_guidance:  # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # following lines are missing:
                # if do_classifier_free_guidance and guidance_rescale > 0.0:
                #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample # compute the previous noisy sample x_t -> x_t-1
                latents = controller.step_callback(latents)  # step callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0): # call the callback, if provided
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0: callback(i, t, latents)
        image = self.decode_latents(latents) # 8. Post-processing
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype) # 9. Run safety checker
        if output_type == "pil": image = self.numpy_to_pil(image) # 10. Convert to PIL
        if not return_dict: return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            # comment Umer: we seem to only using the 2nd attn in each attn block
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet, logger=self.attn_logger
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count

from diffusers.models.cross_attention import CrossAttention


class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet, logger:LoggedVars=None):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet
        self.logger = logger

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        if self.logger is not None:
            self.logger.clear()
            self.logger['attn'] = attn
            self.logger['hidden_states__passed'] = hidden_states
            self.logger['encoder_hidden_states__passed'] = encoder_hidden_states
            self.logger['attention_mask__passed'] = attention_mask
        
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.logger is not None:
            self.logger['batch_size'] = batch_size
            self.logger['sequence_length'] = sequence_length
            self.logger['attn#prepare_attention_mask'] = attn.prepare_attention_mask
            self.logger['attention_mask'] = attention_mask
        
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if self.logger is not None:
            self.logger['query__pre_h2b'] = query
            self.logger['is_cross'] = is_cross
            self.logger['encoder_hidden_states'] = encoder_hidden_states
            self.logger['key__pre_h2b'] = key
            self.logger['value__pre_h2b'] = value

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        if self.logger is not None:
            self.logger['query'] = query
            self.logger['key'] = key
            self.logger['value'] = value
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if self.logger is not None:
            self.logger['attention_probs__precontrol'] = attention_probs
            self.logger['attn#get_attention_scores'] = attn.get_attention_scores
        
        self.controller(attention_probs, is_cross, self.place_in_unet) # one line change
        if self.logger is not None:
            self.logger['attention_probs__postcontrol'] = attention_probs
            self.logger['place_in_unet'] = self.place_in_unet
        
        hidden_states = torch.bmm(attention_probs, value)
        if self.logger is not None: self.logger['hidden_states_1'] = hidden_states
        hidden_states = attn.batch_to_head_dim(hidden_states)
        if self.logger is not None: self.logger['hidden_states_2'] = hidden_states
        hidden_states = attn.to_out[0](hidden_states) # linear proj
        if self.logger is not None: self.logger['hidden_states_3'] = hidden_states
        hidden_states = attn.to_out[1](hidden_states) # dropout
        if self.logger is not None:
            self.logger['hidden_states_4'] = hidden_states
            self.logger['attn#to_out'] = attn.to_out
            self.logger['attn#head_to_batch_dim'] = attn.head_to_batch_dim
            self.logger['attn#batch_to_head_dim'] = attn.batch_to_head_dim
        
        return hidden_states