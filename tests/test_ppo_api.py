import copy
import tempfile
import unittest

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
	DataCollatorForSeq2Seq,
	PreTrainedTokenizerFast,
	Seq2SeqTrainer,
	Seq2SeqTrainingArguments,
	T5Config,
	T5ForConditionalGeneration,
	T5ForSequenceClassification,
)
from trl import (
	AutoModelForSeq2SeqLMWithValueHead,
	PPOConfig,
	PPOTrainer,
	RewardConfig,
	RewardTrainer,
)

from src.phase3_ppo import build_reward_dataset
from src.phase2_lora import build_dataset


class PpoApiSmokeTest(unittest.TestCase):
	def make_tokenizer(self):
		vocabulary = {
			"<pad>": 0,
			"</s>": 1,
			"<unk>": 2,
			"review": 3,
			"summarize": 4,
			"tone": 5,
			"summary": 6,
			"candidate": 7,
			"good": 8,
			"bad": 9,
		}
		backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
		backend.pre_tokenizer = pre_tokenizers.Whitespace()
		backend.post_processor = processors.TemplateProcessing(
			single="$A </s>",
			pair="$A $B:1 </s>:1",
			special_tokens=[("</s>", 1)],
		)
		return PreTrainedTokenizerFast(
			tokenizer_object=backend,
			pad_token="<pad>",
			eos_token="</s>",
			unk_token="<unk>",
		)

	def make_model_config(self, vocab_size):
		return T5Config(
			vocab_size=vocab_size,
			d_model=16,
			d_ff=32,
			num_layers=1,
			num_decoder_layers=1,
			num_heads=2,
			pad_token_id=0,
			eos_token_id=1,
			decoder_start_token_id=0,
			dropout_rate=0.0,
		)

	def test_seq2seq_trainer_completes_one_lora_training_step(self):
		tokenizer = self.make_tokenizer()
		rows = [
			{"review_text": "review", "tone": "neutral", "summary": "good summary"},
			{"review_text": "review tone", "tone": "friendly", "summary": "good summary"},
		]
		dataset = build_dataset(rows, tokenizer, ["neutral", "friendly"], 32, 16)
		model = get_peft_model(
			T5ForConditionalGeneration(self.make_model_config(tokenizer.vocab_size)),
			LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=2, lora_alpha=4, target_modules=["q", "v"]),
		)
		with tempfile.TemporaryDirectory() as output_dir:
			trainer = Seq2SeqTrainer(
				model=model,
				args=Seq2SeqTrainingArguments(
					output_dir=output_dir,
					max_steps=1,
					per_device_train_batch_size=1,
					gradient_accumulation_steps=1,
					gradient_checkpointing=False,
					fp16=False,
					eval_strategy="no",
					save_strategy="no",
					logging_steps=1,
					report_to="none",
				),
				train_dataset=dataset,
				tokenizer=tokenizer,
				data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
			)
			result = trainer.train()
		self.assertEqual(result.global_step, 1)

	def test_seq2seq_lora_policy_completes_one_ppo_step(self):
		tokenizer = self.make_tokenizer()
		model_config = self.make_model_config(tokenizer.vocab_size)
		base_model = T5ForConditionalGeneration(model_config)
		policy = get_peft_model(
			base_model,
			LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=2, lora_alpha=4, target_modules=["q", "v"]),
		)
		policy_with_value_head = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(policy)
		ppo_config = PPOConfig(
			exp_name="tiny-seq2seq-test",
			model_name="tiny-test-model",
			learning_rate=1e-4,
			batch_size=2,
			mini_batch_size=1,
			gradient_accumulation_steps=1,
			ppo_epochs=1,
			steps=1,
			log_with=None,
			remove_unused_columns=False,
		)
		prompt_dataset = Dataset.from_list(
			[
				{"input_ids": [3, 4], "attention_mask": [1, 1]},
				{"input_ids": [4, 5], "attention_mask": [1, 1]},
			]
		)
		trainer = PPOTrainer(
			config=ppo_config,
			model=policy_with_value_head,
			ref_model=copy.deepcopy(policy_with_value_head),
			tokenizer=tokenizer,
			dataset=prompt_dataset,
		)
		batch = next(iter(trainer.dataloader))
		queries = [
			batch["input_ids"][index][batch["attention_mask"][index].bool()]
			for index in range(len(batch["input_ids"]))
		]
		responses = trainer.generate(
			queries,
			batch_size=2,
			return_prompt=False,
			do_sample=False,
			max_new_tokens=3,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)
		stats = trainer.step(queries, responses, [torch.tensor(0.2), torch.tensor(-0.1)])
		self.assertIn("objective/kl", stats)
		self.assertEqual(len(responses), 2)

	def test_reward_trainer_completes_one_lora_training_step(self):
		tokenizer = self.make_tokenizer()
		model_config = self.make_model_config(tokenizer.vocab_size)
		model_config.num_labels = 1
		preferences = [
			{"review_text": "review", "tone": "neutral", "chosen": "good summary", "rejected": "bad summary"},
			{"review_text": "review tone", "tone": "friendly", "chosen": "good summary", "rejected": "bad summary"},
		]
		dataset = build_reward_dataset(
			preferences,
			tokenizer,
			["neutral", "friendly"],
			max_length=32,
		)
		model = T5ForSequenceClassification(model_config)
		trainer = RewardTrainer(
			model=model,
			args=RewardConfig(
				output_dir="unused-test-output",
				num_train_epochs=1,
				per_device_train_batch_size=2,
				gradient_accumulation_steps=1,
				gradient_checkpointing=False,
				fp16=False,
				max_length=32,
				remove_unused_columns=False,
				save_strategy="no",
				logging_steps=1,
				report_to="none",
			),
			train_dataset=dataset,
			tokenizer=tokenizer,
			peft_config=LoraConfig(
				task_type=TaskType.SEQ_CLS,
				r=2,
				lora_alpha=4,
				lora_dropout=0.0,
				target_modules=["q", "v"],
				modules_to_save=["classification_head"],
			),
		)
		result = trainer.train()
		self.assertGreater(result.global_step, 0)


if __name__ == "__main__":
	unittest.main()