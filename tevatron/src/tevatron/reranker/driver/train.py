import logging
import os
import sys

from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import TrainingArguments
from transformers import TrainerCallback, TrainerState, TrainerControl

from tevatron.reranker.arguments import ModelArguments, DataArguments

from tevatron.reranker.modeling import RerankerModel
from tevatron.reranker.dataset import RerankerTrainDataset
from tevatron.reranker.trainer import RerankerTrainer
from tevatron.reranker.collator import RerankerTrainCollator

logger = logging.getLogger(__name__)


class CheckpointCleanupCallback(TrainerCallback):
    """Xóa optimizer/scheduler states sau mỗi checkpoint để tiết kiệm disk.

    Mỗi checkpoint-N/ giữ lại đúng những file cần cho inference:
      - adapter_config.json       (~1 KB)
      - adapter_model.safetensors (~39 MB với Qwen3-0.6B)
      - trainer_state.json        (training logs, giữ để debug)

    Những thứ bị xóa (~78 MB/checkpoint):
      - optimizer.pt    — AdamW optimizer states (không cần cho inference)
      - rng_state.pth   — trạng thái random number generator
      - scheduler.pt    — trạng thái learning rate scheduler
      - training_args.bin
      - README.md       — được PEFT tự sinh, không cần
    """

    _RESUME_FILES = [
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "training_args.bin",
        "README.md",
    ]

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return

        freed = 0
        for fname in self._RESUME_FILES:
            fpath = os.path.join(ckpt_dir, fname)
            if os.path.isfile(fpath):
                freed += os.path.getsize(fpath)
                os.remove(fpath)

        remaining = sum(
            os.path.getsize(os.path.join(ckpt_dir, f))
            for f in os.listdir(ckpt_dir)
            if os.path.isfile(os.path.join(ckpt_dir, f))
        )
        logger.info(
            f"[CheckpointCleanup] {ckpt_dir} cleaned — "
            f"freed: {freed / 1024 / 1024:.1f} MB, "
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
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = 'right'
    model = RerankerModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        num_labels=1,
        attn_implementation="flash_attention_2",
    )

    train_dataset = RerankerTrainDataset(data_args)
    train_collator = RerankerTrainCollator(data_args, tokenizer)

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        callbacks=[CheckpointCleanupCallback()],
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
