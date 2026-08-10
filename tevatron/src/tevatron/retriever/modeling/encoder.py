from dataclasses import dataclass
from typing import Dict, Optional

import os

import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import PreTrainedModel, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from tevatron.retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments

import logging
logger = logging.getLogger(__name__)


def _mark_all_lora_adapters_trainable(peft_model: PeftModel) -> None:
    """Force requires_grad=True on every LoRA parameter, across all attached adapters.

    Needed for untied encoders: PEFT's own bookkeeping (`_mark_only_adapters_as_trainable`,
    re-run internally by `add_adapter`/`load_adapter`) only marks the *currently active*
    adapter as trainable and silently freezes the others — so building 'query' then
    attaching 'passage' leaves 'query' with requires_grad=False. Since the base backbone
    is frozen by construction (only `lora_` parameters ever exist as trainable here), it's
    safe to just re-enable grad on every `lora_` parameter regardless of which one is
    "active" at this moment.
    """
    for name, param in peft_model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 untie_encoder: bool = False,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.untie_encoder = untie_encoder
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _set_adapter(self, name: str):
        """Switch the active LoRA adapter before encoding query vs. passage.

        No-op when the encoder is tied (single/default adapter, `untie_encoder=False`)
        or when `self.encoder` isn't a PEFT model at all (e.g. full fine-tuning without
        `--lora`). This is what lets an "untied" bi-encoder share one frozen backbone
        instead of loading two full copies of it: only the (tiny) LoRA adapter differs
        between the query-encoding and passage-encoding forward pass.
        """
        if self.untie_encoder and hasattr(self.encoder, 'set_adapter'):
            self.encoder.set_adapter(name)

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        if self.untie_encoder and q_reps is not None and p_reps is not None:
            # PEFT's set_adapter() (called by _set_adapter above, once per encode_*
            # call) always flips requires_grad=False on whichever adapter is NOT the
            # one just activated — a documented PEFT side effect, not something we
            # opted into. Since encode_passage() runs after encode_query(), by this
            # point 'query' has requires_grad=False again, even though q_reps was
            # correctly computed with the query adapter above. Both adapters feed
            # into the single joint loss below, so re-enable both now, before
            # backward() (called later, outside this function) runs.
            _mark_all_lora_adapters_trainable(self.encoder)

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores / self.temperature, target)
            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
                if model_args.untie_encoder:
                    passage_path = os.path.join(model_args.lora_name_or_path, 'passage')
                    if os.path.isdir(passage_path):
                        lora_model.load_adapter(passage_path, adapter_name='passage', is_trainable=True)
                        _mark_all_lora_adapters_trainable(lora_model)
            elif model_args.untie_encoder:
                query_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                passage_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                # Two independently-initialized LoRA adapters on top of ONE shared frozen
                # backbone: `add_adapter` attaches a second set of low-rank matrices to the
                # same base weights rather than instantiating a second copy of the model.
                lora_model = get_peft_model(base_model, query_config, adapter_name='query')
                lora_model.add_adapter('passage', passage_config)
                _mark_all_lora_adapters_trainable(lora_model)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                untie_encoder=model_args.untie_encoder,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             untie_encoder: bool = False,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            if untie_encoder:
                # Keep both adapters attached to the single shared backbone rather than
                # merge_and_unload(): merging would require materializing two separate
                # full-weight copies of the backbone (one per adapter), which defeats the
                # whole point of untying via adapters instead of via duplicate models.
                # encode_query/encode_passage switch the active adapter at call time.
                query_path = os.path.join(lora_name_or_path, 'query')
                passage_path = os.path.join(lora_name_or_path, 'passage')
                lora_model = PeftModel.from_pretrained(base_model, query_path, adapter_name='query')
                lora_model.load_adapter(passage_path, adapter_name='passage')
                model = cls(
                    encoder=lora_model,
                    pooling=pooling,
                    normalize=normalize,
                    untie_encoder=True,
                )
            else:
                lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
                lora_model = lora_model.merge_and_unload()
                model = cls(
                    encoder=lora_model,
                    pooling=pooling,
                    normalize=normalize
                )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
