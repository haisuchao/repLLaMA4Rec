import glob
import logging
import os
import shutil
import sys
import torch
from transformers import AutoTokenizer, TrainerCallback, TrainerControl, TrainerState
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.collator import TrainCollator
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.trainer import TevatronTrainer as Trainer
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer

logger = logging.getLogger(__name__)


class CheckpointCleanupCallback(TrainerCallback):
    """Xóa DeepSpeed optimizer/model states sau mỗi checkpoint để tiết kiệm disk.

    Mỗi checkpoint-N/ giữ lại đúng 2 file cần cho eval:
      - adapter_config.json       (~1 KB)
      - adapter_model.safetensors (~20 MB với Qwen3-0.6B)

    Những thứ bị xóa (~2.4 GB/checkpoint):
      - global_step*/   — DeepSpeed model + optimizer states (chỉ cần để resume)
      - rng_state.pth   — trạng thái random number generator
      - scheduler.pt    — trạng thái learning rate scheduler
      - training_args.bin
      - zero_to_fp32.py
    """

    _RESUME_FILES = [
        "rng_state.pth",
        "scheduler.pt",
        "training_args.bin",
        "zero_to_fp32.py",
    ]

    def on_save(
        self,
        args: "HFTrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return

        # Xóa global_step*/ — phần lớn nhất (~2.3 GB DeepSpeed model state)
        for path in glob.glob(os.path.join(ckpt_dir, "global_step*")):
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"[CheckpointCleanup] Removed DeepSpeed states: {path}")

        # Xóa các file chỉ cần để resume training
        for fname in self._RESUME_FILES:
            fpath = os.path.join(ckpt_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)

        remaining = sum(
            os.path.getsize(os.path.join(ckpt_dir, f))
            for f in os.listdir(ckpt_dir)
            if os.path.isfile(os.path.join(ckpt_dir, f))
        )
        logger.info(
            f"[CheckpointCleanup] {ckpt_dir} cleaned — "
            f"remaining: {remaining / 1024 / 1024:.1f} MB"
        )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.padding_side == 'right':
        tokenizer.padding_side = 'right'
    else:
        tokenizer.padding_side = 'left'
    
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args, tokenizer)

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[CheckpointCleanupCallback()],
    )
    train_dataset.set_trainer(trainer)
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
