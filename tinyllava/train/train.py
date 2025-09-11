from packaging import version
import pathlib

import tokenizers
import transformers
from peft import PeftModel
from tinyllava.utils import get_state_maybe_zero_3
from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import ModelArguments, DataArguments, TrainingArguments, get_peft_state_maybe_zero_3, logger_setting, log_trainable_params
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

from transformers import (
    TrainerCallback
)

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args



class SaveCallback(TrainerCallback):
    def __init__(self, model, tokenizer, training_recipe):
        self.model = model
        self.tokenizer = tokenizer
        self.training_recipe = training_recipe
        
    def on_save(self, args, state, control, model=None, **kwargs):
        if hasattr(control, "output_dir") and control.output_dir is not None:
            checkpoint_dir = control.output_dir
        else:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        #save language model
        model = self.model
        language_model_state_dict = get_state_maybe_zero_3(model.language_model.named_parameters(), [''], False)
        if args.local_rank == 0 or args.local_rank == -1:
            language_model_output_dir = os.path.join(checkpoint_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)
            language_model_output_path = os.path.join(checkpoint_dir, 'language_model/pytorch_model.bin')
            torch.save(language_model_state_dict, language_model_output_path)
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)
        #save vision tower
        vision_tower_state_dict = get_state_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), [''], False)
        if args.local_rank == 0 or args.local_rank == -1:
            vision_tower_output_dir = os.path.join(checkpoint_dir, 'vision_tower')
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            vision_tower_output_path = os.path.join(checkpoint_dir, 'vision_tower/pytorch_model.bin')
            torch.save(vision_tower_state_dict, vision_tower_output_path)
            if isinstance(model.vision_tower._vision_tower, PreTrainedModel):
                model.vision_tower._vision_tower.config.save_pretrained(vision_tower_output_dir, from_pt=True)
        #save connector
        connector_state_dict = get_state_maybe_zero_3(model.connector.named_parameters(), [''], False)
        if args.local_rank == 0 or args.local_rank == -1:
            connector_output_dir = os.path.join(checkpoint_dir, 'connector')
            os.makedirs(connector_output_dir, exist_ok=True)
            connector_output_path = os.path.join(checkpoint_dir, 'connector/pytorch_model.bin')
            torch.save(connector_state_dict, connector_output_path)

        if "lora" in self.training_recipe:
            if args.local_rank == 0 or args.local_rank == -1:
                if isinstance(model, PeftModel):
                    base_model = model.merge_and_unload()  # 把 LoRA 合并回原模型
                    base_model.save_pretrained(checkpoint_dir)
                else:
                    model.save_pretrained(checkpoint_dir)
                # model.save_pretrained(checkpoint_dir, state_dict=lora_state_dict)

def train():
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = TinyLlavaForConditionalGeneration(model_config)
    # load pretrained checkpoint
    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])

    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3

    callbacks = []
    callbacks.append(SaveCallback(model, tokenizer, training_arguments.training_recipe))
    trainer = LLaVATrainer(model=model, # does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           callbacks = callbacks, 
                           args=training_arguments,
                           **data_module)
    
    trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
