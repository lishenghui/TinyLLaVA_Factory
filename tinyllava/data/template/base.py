from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import copy, re

from .formatter import EmptyFormatter, StringFormatter
from .formatter import Formatter
from ...utils.constants import *
from .utils_template import (
    parse_positions,
    get_intervention_locations,
    get_intervention_locations_uniform,
    replace_pad_in_2d_list_intervention_locations,
    parse_positions_uniform)

from transformers import PreTrainedTokenizer
import torch
    

def add_integer_to_2d_list(lst, integer):
    return [[(element + integer) if element != -1 else -1 for element in sublist] for sublist in lst]

def add_integer_to_list(lst, integer):
    return [(element + integer) if element != -1 else -1 for element in lst]


@dataclass
class Template:
    format_image_token: "Formatter"
    format_user: "Formatter"
    format_assistant: "Formatter"
    system: "Formatter"
    separator: "Formatter"
    mores_pos_configs: Union[Dict, List[Dict]] = None

    def _concat_two_intervention_locations(self, intervention_locations, intervention_locations_2) -> Tuple[List[List], int, int]:
        assert len(intervention_locations) == len(intervention_locations_2), "The length of two intervention_locations doesn't equal, both should be the number of interventions."
        return ([intervention_locations[i] + intervention_locations_2[i] for i in range(len(intervention_locations))],
                len(intervention_locations[0]),
                len(intervention_locations_2[0]))

    
    def get_intervention_locations_for_multimodal_prompt(self, question_prompt, tokenizer, question_content):
        if isinstance(self.mores_pos_configs, Dict):
            return self._get_intervention_locations_for_multimodal_prompt(question_prompt, tokenizer, question_content, self.mores_pos_configs)
        elif isinstance(self.mores_pos_configs, List) and isinstance(self.mores_pos_configs[0], Dict):
            intervention_locations = self._get_intervention_locations_for_multimodal_prompt(question_prompt, tokenizer, question_content, self.mores_pos_configs[0])
            intervention_locations_2 = self._get_intervention_locations_for_multimodal_prompt(question_prompt, tokenizer, question_content, self.mores_pos_configs[1])
            return self._concat_two_intervention_locations(intervention_locations, intervention_locations_2)
        else:
            raise ValueError("invalid format of mores_pos_configs in Template")

    def _get_intervention_locations_for_multimodal_prompt(self, question_prompt, tokenizer, question_content, mores_pos_configs):
        # intervention locations
        question_prompt_ids = self.tokenizer_image_token(question_prompt, tokenizer, return_tensors='pt')

        img_token_position = torch.where(question_prompt_ids == IMAGE_TOKEN_INDEX)[0]
        num_img_token = len(img_token_position)

        question_prompt_length = len(question_prompt_ids)

        question_content_ids = self.tokenizer_image_token(question_content, tokenizer, return_tensors='pt')
        question_content_length = len(question_content_ids)
        sys_length = question_prompt_length - question_content_length
        
        if bool(re.fullmatch(r'^f\d+\++l\d+$', mores_pos_configs['intervention_positions'])):
            intervention_locations, start_pos, end_pos, intervention_part_len = self._intervention_locations_fl(sys_length,
                                                                                                                question_prompt_length,
                                                                                                                num_img_token,
                                                                                                                mores_pos_configs)
        elif bool(re.fullmatch(r'^uniform\d+$', mores_pos_configs['intervention_positions'])):
            intervention_locations, start_pos, end_pos, intervention_part_len = self._intervention_locations_uniform(sys_length,
                                                                                                                     question_prompt_length,
                                                                                                                     num_img_token,
                                                                                                                     mores_pos_configs)
        else:
            raise ValueError("Get invalid intervention_positions! It should be like either \"f4+l5\" or \"uniform9\"!")
        
        intervention_locations = add_integer_to_2d_list(intervention_locations, start_pos)
        if all(x == -1 for x in intervention_locations[0]) and intervention_part_len == 1:
            if "text" in mores_pos_configs['intervene_modality']:   # condition 1
                intervention_locations = [[start_pos] + [-1] * (len(intervention_locations[0])-1)] * len(intervention_locations)
            else:   # condition 2
                intervention_locations = [[start_pos - 1] + [-1] * (len(intervention_locations[0])-1)] * len(intervention_locations)
                assert start_pos - 1 >= 0, "There's no (sys) token in front of the text tokens!"
        replace_pad_in_2d_list_intervention_locations(intervention_locations)
        return intervention_locations
    
    def _intervention_locations_fl(self, sys_length, question_prompt_length, num_img_token, mores_pos_configs):
        first_n, last_n = parse_positions(mores_pos_configs['intervention_positions'])
        img_embed_token_len = mores_pos_configs['img_embed_token_len']

        assert "text" in mores_pos_configs['intervene_modality'] or "vis" in mores_pos_configs['intervene_modality'], "intervene_modality has to at least include one of \"vis\" and \"text\""
        if "text" in mores_pos_configs['intervene_modality']:
            end_pos = question_prompt_length - 1 + (img_embed_token_len-1)*num_img_token
        else:
            end_pos = sys_length - 1 + img_embed_token_len*1 if num_img_token != 0 else sys_length
        if "vis" in mores_pos_configs['intervene_modality']:
            start_pos = sys_length
        else:
            start_pos = sys_length + (img_embed_token_len)*1 if num_img_token != 0 else sys_length

        intervention_part_len = end_pos - start_pos + 1   
        # two conditions lead to 1: (1) only 1 token when intervening only on text (2) no image token when intervening only on vis
        assert intervention_part_len >= 1, "There's no question content (zero tokens) to interevene!" 
        intervention_locations = get_intervention_locations(
            last_position=intervention_part_len, 
            first_n=first_n,
            last_n=last_n,
            pad_mode="first",
            share_weights=mores_pos_configs['mores_share_weights'],
            num_interventions=mores_pos_configs['num_interventions']['llm']
        )
        return intervention_locations, start_pos, end_pos, intervention_part_len
    
    def _intervention_locations_uniform(self, sys_length, question_prompt_length, num_img_token, mores_pos_configs):
        num_uni_samp = parse_positions_uniform(mores_pos_configs['intervention_positions'])
        img_embed_token_len = mores_pos_configs['img_embed_token_len']

        assert "vis" in mores_pos_configs['intervene_modality'], "For uniform sampled positions, intervene_modality has to be \"vis\", which has the fixed length of token"

        start_pos = sys_length
        end_pos = sys_length - 1 + img_embed_token_len*1 if num_img_token != 0 else sys_length

        intervention_part_len = end_pos - start_pos + 1   
        # two conditions lead to 1: (1) only 1 token when intervening only on text (2) no image token when intervening only on vis
        assert intervention_part_len >= 1, "There's no question content (zero tokens) to interevene!" 
        intervention_locations = get_intervention_locations_uniform(
            last_position=intervention_part_len, 
            num_uni_samp=num_uni_samp,
            pad_mode="first",
            share_weights=mores_pos_configs['mores_share_weights'],
            num_interventions=mores_pos_configs['num_interventions']['llm']
        )
        return intervention_locations, start_pos, end_pos, intervention_part_len

    def get_sys_len(self, question_prompt, tokenizer, question_content):
        def find_exact_subtensor(main_tensor, sub_tensor):
            """
            Find the starting index of an exact sub_tensor within main_tensor.

            Parameters:
            - main_tensor (torch.Tensor): 1D tensor.
            - sub_tensor (torch.Tensor): 1D tensor to find within main_tensor.

            Returns:
            - int: Starting index if found, else -1.
            """
            main_length = len(main_tensor)
            sub_length = len(sub_tensor)

            if sub_length > main_length:
                return -1  # Sub-tensor is longer than main tensor

            for i in range(main_length - sub_length + 1):
                # Slice the main tensor
                segment = main_tensor[i:i + sub_length]
                if torch.equal(segment, sub_tensor):
                    return i  # Match found

            return -1  # No match found

        question_prompt_ids = self.tokenizer_image_token(question_prompt, tokenizer, return_tensors='pt')
        question_content_ids = self.tokenizer_image_token(question_content, tokenizer, return_tensors='pt')
        ques_start_idx = find_exact_subtensor(question_prompt_ids, question_content_ids)

        return ques_start_idx

    def encode(self, messages, tokenizer, mode='train', return_sys_len=False):
        """
        1. get list form messages(conversations:[{from:human, value:message}, {from:gpt, value:message}])
            ===>  human_list, value_list
        2. prompt two list
        3. tokenize prompt
        4. make target
        """
        question_list, answer_list = self.get_list_from_message(messages, mode)
        prompt, question_prompt, question_content = self.prompt(question_list, answer_list, return_prompt=True)
        input_ids = self.tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
        sys_len = self.get_sys_len(question_prompt, tokenizer, question_content) if return_sys_len else None
        
        len_two_pos_configs = None
        intervention_locations = None
        if self.mores_pos_configs != None:
            if isinstance(self.mores_pos_configs, Dict):
                intervention_locations = self.get_intervention_locations_for_multimodal_prompt(question_prompt, tokenizer, question_content)
            elif isinstance(self.mores_pos_configs, List) and isinstance(self.mores_pos_configs[0], Dict):
                intervention_locations, len_pos_1, len_pos_2 = self.get_intervention_locations_for_multimodal_prompt(question_prompt, tokenizer, question_content)
                len_two_pos_configs = (len_pos_1, len_pos_2)

        if mode == 'train':
            labels = self.make_labels(input_ids, prompt, tokenizer)
            return dict(
                input_ids=input_ids,
                labels=labels,
                intervention_locations=intervention_locations,
                len_two_pos_configs=len_two_pos_configs,
                sys_len=sys_len
            )
        else:
            return dict(
                input_ids=input_ids,
                prompt=prompt,
                intervention_locations=intervention_locations,
                len_two_pos_configs=len_two_pos_configs,
                sys_len=sys_len
            )
        
    
    def get_list_from_message(self, messages, mode='train'):
        return self._get_list_from_message(messages, mode)
    
    def _get_list_from_message(self, messages, mode='train'):
        """
        messages  ====>  [{from:human, value:message}, {from:gpt, value:message}]
        """
        question_list = []
        answer_list = []
        first_is_not_question = 0
        for i, message in enumerate(messages):
            if i == 0 and message['from'] != 'human':
                first_is_not_question = 1
                continue
            if i % 2 == first_is_not_question:
                question_list.append(message['value'])
            else:
                answer_list.append(message['value'])
        if mode=='train':
            assert len(question_list) == len(answer_list) , \
                f"qa is not match : length_q:{len(question_list)} vs length_a:{len(answer_list)}"
        return question_list, answer_list
    

    def prompt(
        self,
        question_list, answer_list,
        return_prompt: bool = False
    ):
        if type(question_list) is str:
            question_list = [question_list]
        if type(answer_list) is str:
            answer_list = [answer_list]    
        msg, question_prompt, question_content = self._prompt(question_list, answer_list) if len(answer_list) == len(question_list) else self._prompt_only_q(question_list)
        if not return_prompt:
            return msg
        else:
            return msg, question_prompt, question_content

    def _prompt(
        self,
        question_list, answer_list,
    ):
        msg = ""
        question_prompt = ""
        question_content = None
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += self.system.apply()
            if DEFAULT_IMAGE_TOKEN in question:
                question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                question = self.format_image_token.apply(content=question).strip()
            msg += self.format_user.apply(content=question)

            if i == 0:
                question_prompt = msg
                question_content = question

            msg += self.format_assistant.apply(content=answer)
        return msg, question_prompt, question_content
    
    def _prompt_only_q(
        self,
        question_list
    ):
        msg = ""
        question_prompt = ""
        question_content = None
        for i, question in enumerate(question_list):
            if i == 0:
                msg += self.system.apply()
            if DEFAULT_IMAGE_TOKEN in question:
                question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                question = self.format_image_token.apply(content=question).strip()
            msg += self.format_user.apply(content=question)

            if i == 0:
                question_prompt = msg
                question_content = question
        return msg, question_prompt, question_content
    
    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        sep, eos_token = self.separator.apply()
        total_len = int(labels.ne(tokenizer.pad_token_id).sum())
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            total_len += prompt.count(eos_token)
        rounds = prompt.split(eos_token)
        eos_token_length = len(tokenizer.encode(eos_token))
        labels, cur_len = self._make_masks(labels, tokenizer, sep, eos_token_length, rounds)
        if cur_len < tokenizer.model_max_length:
            import time
            if cur_len != total_len:
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print("number of rounds: ", len(rounds) - 1)
                print("rounds: ", rounds[:-1])
                print("prompt: ", prompt)
                print(labels)
                print(input_ids)
                time.sleep(5)
                labels[:] = IGNORE_INDEX
        return labels
        
        
        
    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 0
        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len
        
    @classmethod    
    def tokenizer_image_token(cls, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        def _insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in _insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids





