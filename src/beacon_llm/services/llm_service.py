from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, TrainingArguments, \
    DataCollatorForLanguageModeling, Trainer, AutoModelForCausalLM


class LLMService:
    def inference(self, model_name, prompt):
        raise NotImplementedError("inference is not yet implemented")

    def fine_tune(self, model_name, data_dir_path) -> bool:
        # ───────────────────────────────────────────────
        #  Model setup
        # ───────────────────────────────────────────────

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype="auto",
                                                     trust_remote_code=True, low_cpu_mem_usage=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=2,
            lora_alpha=4,  # r * 2
            lora_dropout=0.1,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            modules_to_save=None,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ──────────────────────────────────────────────
        #  Dataset setup
        # ───────────────────────────────────────────────

        train_dataset = load_dataset("csv",
                                     data_files={"train": str(
                                         Path(__file__).resolve().parent.parent / "data" / "dataset" / "train.csv")},
                                     split="train")

        def tokenize_function(data):
            return tokenizer(
                data["sentence"],
                truncation=True,
                max_length=64,
                padding='max_length'
            )

        tokenzied_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=2,
                                                    remove_columns=train_dataset.column_names)

        data_collator_config = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # ───────────────────────────────────────────────
        #  Training setup
        # ───────────────────────────────────────────────

        train_args = TrainingArguments(
            output_dir=str(Path.cwd() / "out"),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            learning_rate=2e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            logging_strategy="steps",
            logging_steps=500,
            fp16=True,
            bf16=False,
            optim="adafactor",
            eval_strategy="no",
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            report_to="none",
            max_grad_norm=0.3,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenzied_train_dataset,
            processing_class=tokenizer,
            data_collator=data_collator_config,
        )

        trainer.train()

        # ───────────────────────────────────────────────
        #  Save Model
        # ───────────────────────────────────────────────

        model.save_pretrained(train_args.output_dir)
        tokenizer.save_pretrained(train_args.output_dir)

        return True

    def export_to_gguf(self):
        raise NotImplementedError("export_to_gguf is not yet implemented")
