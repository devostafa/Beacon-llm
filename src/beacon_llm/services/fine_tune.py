from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, \
    DataCollatorForLanguageModeling, Trainer


class FineTune:
    def fine_tune(model_name, data_dir_path) -> bool:
        # ───────────────────────────────────────────────
        #  Model setup
        # ───────────────────────────────────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, dtype="auto",
                                                     device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )

        model = get_peft_model(model, lora_config)

        # ──────────────────────────────────────────────
        #  Dataset setup
        # ───────────────────────────────────────────────

        train_dataset = load_dataset("csv", data_dir=data_dir_path, split="train")

        def tokenize_function(data):
            return tokenizer(
                data["text"],
                truncation=True,
                max_length=1024,
                padding=False,
            )

        tokenzied_train_dataset = train_dataset.map(tokenize_function, batched=True,
                                                    remove_columns=train_dataset["train"].column_names)

        data_collator_config = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # ───────────────────────────────────────────────
        #  Training setup
        # ───────────────────────────────────────────────

        train_args = TrainingArguments(
            output_dir=str(Path(__file__).resolve().parent.parent.parent / "out"),
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            bf16=True,
            learning_rate=2e-5,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
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

        model = model.merge_and_unload()

        model.save_pretrained(train_args.output_dir)

        return true
