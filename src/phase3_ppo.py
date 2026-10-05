"""Train a preference reward model, then align the SFT LoRA adapter with PPO."""

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import (
	AutoModelForSeq2SeqLMWithValueHead,
	PPOConfig,
	PPOTrainer,
	RewardConfig,
	RewardTrainer,
)

from src.data import read_annotation_csv
from src.phase1_prompting import build_prompt, load_config


PREFERENCE_COLUMNS = ("review_text", "tone", "chosen", "rejected")


def resolve_path(project_root: Path, value: str | Path) -> Path:
	path = Path(value)
	return path if path.is_absolute() else project_root / path


def validate_preferences(rows: list[dict[str, str]], tones: list[str]) -> None:
	invalid_tones = sorted({row["tone"] for row in rows} - set(tones))
	if invalid_tones:
		raise ValueError(f"Tonos no definidos en configs/default.yaml: {', '.join(invalid_tones)}")
	identical_pairs = [index for index, row in enumerate(rows, start=2) if row["chosen"] == row["rejected"]]
	if identical_pairs:
		raise ValueError(f"chosen y rejected son iguales en las filas: {identical_pairs}")


def build_reward_dataset(
	rows: list[dict[str, str]],
	tokenizer: Any,
	tones: list[str],
	max_length: int,
) -> Dataset:
	"""Tokenize preferred/rejected completions in TRL's expected pair format."""
	validate_preferences(rows, tones)
	tokenized_pairs = []
	for row in rows:
		prompt = build_prompt(row["review_text"], row["tone"])
		chosen = tokenizer(
			f"{prompt}\nCandidate summary:\n{row['chosen']}",
			max_length=max_length,
			truncation=True,
		)
		rejected = tokenizer(
			f"{prompt}\nCandidate summary:\n{row['rejected']}",
			max_length=max_length,
			truncation=True,
		)
		tokenized_pairs.append(
			{
				"input_ids_chosen": chosen["input_ids"],
				"attention_mask_chosen": chosen["attention_mask"],
				"input_ids_rejected": rejected["input_ids"],
				"attention_mask_rejected": rejected["attention_mask"],
			}
		)
	return Dataset.from_list(tokenized_pairs)


def train_reward_model(config: dict[str, Any], preference_file: Path, output_dir: Path) -> None:
	phase_config = config["phase3"]
	rows = read_annotation_csv(preference_file, PREFERENCE_COLUMNS)
	validate_preferences(rows, config["phase1"]["tone_labels"])
	if len(rows) < 2:
		raise ValueError("Se necesitan al menos dos comparaciones humanas para entrenar el reward model.")
	print(f"Pares de preferencia anotados: {len(rows)}")

	model_name = config["project"]["model_name"]
	device_has_cuda = torch.cuda.is_available()
	dtype = torch.float32
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	dataset = build_reward_dataset(
		rows,
		tokenizer,
		config["phase1"]["tone_labels"],
		phase_config["reward_max_length"],
	)
	eval_dataset = None
	if len(dataset) >= 10:
		split = dataset.train_test_split(test_size=0.1, seed=config["training"]["seed"])
		dataset, eval_dataset = split["train"], split["test"]

	reward_model = AutoModelForSequenceClassification.from_pretrained(
		model_name,
		num_labels=1,
		torch_dtype=dtype,
	)
	reward_model.config.use_cache = False
	lora_config = LoraConfig(
		task_type=TaskType.SEQ_CLS,
		r=phase_config["lora_r"],
		lora_alpha=phase_config["lora_alpha"],
		lora_dropout=phase_config["lora_dropout"],
		target_modules=phase_config["target_modules"],
		modules_to_save=["classification_head"],
		bias="none",
	)
	reward_args = RewardConfig(
		output_dir=str(output_dir),
		max_length=phase_config["reward_max_length"],
		num_train_epochs=phase_config["reward_num_train_epochs"],
		learning_rate=phase_config["reward_learning_rate"],
		per_device_train_batch_size=phase_config["reward_batch_size"],
		per_device_eval_batch_size=phase_config["reward_batch_size"],
		gradient_accumulation_steps=phase_config["reward_gradient_accumulation_steps"],
		fp16=False,
		gradient_checkpointing=False,
		eval_strategy="epoch" if eval_dataset else "no",
		save_strategy="epoch",
		save_total_limit=2,
		logging_steps=5,
		report_to="none",
		remove_unused_columns=False,
		seed=config["training"]["seed"],
	)
	trainer = RewardTrainer(
		model=reward_model,
		args=reward_args,
		train_dataset=dataset,
		eval_dataset=eval_dataset,
		tokenizer=tokenizer,
		peft_config=lora_config,
	)
	trainer.train()
	trainer.save_model(str(output_dir))
	tokenizer.save_pretrained(output_dir)
	print(f"Adaptador reward LoRA guardado en {output_dir}")


def score_candidate(
	prompt: str,
	response: str,
	tokenizer: Any,
	reward_model: Any,
	device: torch.device,
	max_length: int,
) -> torch.Tensor:
	encoded = tokenizer(
		f"{prompt}\nCandidate summary:\n{response}",
		return_tensors="pt",
		max_length=max_length,
		truncation=True,
	).to(device)
	with torch.inference_mode():
		return reward_model(**encoded).logits.reshape(-1)[0].float()


def build_prompt_dataset(
	rows: list[dict[str, str]],
	tokenizer: Any,
	tones: list[str],
	max_input_tokens: int,
) -> Dataset:
	validate_preferences(rows, tones)
	unique_prompts = list(dict.fromkeys(build_prompt(row["review_text"], row["tone"]) for row in rows))
	examples = []
	for prompt in unique_prompts:
		encoded = tokenizer(
			prompt,
			padding="max_length",
			max_length=max_input_tokens,
			truncation=True,
			return_tensors="pt",
		)
		examples.append(
			{
				"input_ids": encoded["input_ids"].squeeze(0).tolist(),
				"attention_mask": encoded["attention_mask"].squeeze(0).tolist(),
			}
		)
	return Dataset.from_list(examples)


def train_ppo(
	config: dict[str, Any],
	preference_file: Path,
	policy_adapter_dir: Path,
	reward_adapter_dir: Path,
	output_dir: Path,
) -> None:
	phase_config = config["phase3"]
	rows = read_annotation_csv(preference_file, PREFERENCE_COLUMNS)
	validate_preferences(rows, config["phase1"]["tone_labels"])
	if not (policy_adapter_dir / "adapter_config.json").is_file():
		raise FileNotFoundError(
			f"No encuentro un adaptador LoRA en {policy_adapter_dir}. Ejecutá primero la Fase 2."
		)
	if not (reward_adapter_dir / "adapter_config.json").is_file():
		raise FileNotFoundError(
			f"No encuentro un reward model LoRA en {reward_adapter_dir}. "
			"Ejecutá primero `phase3_ppo.py reward`."
		)

	model_name = config["project"]["model_name"]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dtype = torch.float16 if device.type == "cuda" else torch.float32
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
		str(policy_adapter_dir),
		is_trainable=True,
		torch_dtype=dtype,
	).to(device)
	ref_policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
		str(policy_adapter_dir),
		is_trainable=False,
		torch_dtype=dtype,
	).to(device)
	policy.config.pad_token_id = tokenizer.pad_token_id
	ref_policy.config.pad_token_id = tokenizer.pad_token_id
	reward_base = AutoModelForSequenceClassification.from_pretrained(
		model_name,
		num_labels=1,
		torch_dtype=dtype,
	)
	reward_model = PeftModel.from_pretrained(reward_base, str(reward_adapter_dir)).to(device)
	reward_model.eval()

	ppo_dataset = build_prompt_dataset(
		rows,
		tokenizer,
		config["phase1"]["tone_labels"],
		phase_config["ppo_max_input_tokens"],
	)
	if len(ppo_dataset) < 2:
		raise ValueError("PPO necesita al menos dos prompts diferentes para formar un batch.")
	ppo_config = PPOConfig(
		exp_name="flan-t5-controlled-tone",
		model_name=model_name,
		learning_rate=phase_config["ppo_learning_rate"],
		batch_size=2,
		mini_batch_size=1,
		gradient_accumulation_steps=1,
		ppo_epochs=phase_config["ppo_epochs"],
		steps=phase_config["ppo_max_steps"],
		init_kl_coef=phase_config["kl_coefficient"],
		remove_unused_columns=False,
		log_with=None,
		seed=config["training"]["seed"],
	)
	trainer = PPOTrainer(
		config=ppo_config,
		model=policy,
		ref_model=ref_policy,
		tokenizer=tokenizer,
		dataset=ppo_dataset,
	)
	generation_kwargs = {
		"do_sample": True,
		"top_k": 50,
		"top_p": 0.95,
		"temperature": 0.8,
		"max_new_tokens": phase_config["ppo_max_new_tokens"],
		"pad_token_id": tokenizer.pad_token_id,
		"eos_token_id": tokenizer.eos_token_id,
	}
	output_dir.mkdir(parents=True, exist_ok=True)
	metrics_path = output_dir / "ppo_metrics.csv"
	step_count = 0
	prompt_pool = [build_prompt(row["review_text"], row["tone"]) for row in rows]
	with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
		writer = csv.DictWriter(metrics_file, fieldnames=["step", "reward", "kl"])
		writer.writeheader()
		for _epoch in range(phase_config["ppo_num_epochs"]):
			for _batch_index in range(0, len(prompt_pool), 2):
				if step_count >= phase_config["ppo_max_steps"]:
					break
				batch_prompts = prompt_pool[_batch_index : _batch_index + 2]
				if not batch_prompts:
					continue
				encoded_prompts = [
					tokenizer(
						prompt,
						max_length=phase_config["ppo_max_input_tokens"],
						truncation=True,
						return_tensors="pt",
					)
					for prompt in batch_prompts
				]
				query_tensors = [
					encoded["input_ids"][0].to(device)[encoded["attention_mask"][0].bool()]
					for encoded in encoded_prompts
				]
				prompts = [tokenizer.decode(query, skip_special_tokens=True) for query in query_tensors]
				padded_inputs = torch.nn.utils.rnn.pad_sequence(
					query_tensors,
					batch_first=True,
					padding_value=tokenizer.pad_token_id,
				)
				attention_mask = torch.nn.utils.rnn.pad_sequence(
					[
						torch.ones(len(query), device=device, dtype=torch.long)
						for query in query_tensors
					],
					batch_first=True,
					padding_value=0,
				)
				response_batch = policy.generate(
					input_ids=padded_inputs,
					attention_mask=attention_mask,
					**generation_kwargs,
				)
				response_batch = [response_batch[index] for index in range(response_batch.shape[0])]
				response_texts = [
					tokenizer.decode(response, skip_special_tokens=True).strip()
					for response in response_batch
				]
				rewards = [
					score_candidate(
						prompt,
						response,
						tokenizer,
						reward_model,
						device,
						phase_config["reward_max_length"],
					)
					for prompt, response in zip(prompts, response_texts)
				]
				stats = trainer.step(query_tensors, response_batch, rewards)
				step_count += 1
				writer.writerow(
					{
						"step": step_count,
						"reward": float(torch.stack(rewards).mean().detach().cpu()),
						"kl": float(stats["objective/kl"]),
					}
				)
				metrics_file.flush()
				print(f"Paso PPO {step_count}: reward={float(torch.stack(rewards).mean()):.4f}")
			if step_count >= phase_config["ppo_max_steps"]:
				break

		trained_policy = trainer.accelerator.unwrap_model(trainer.model)
		trained_policy.save_pretrained(str(output_dir))
		tokenizer.save_pretrained(output_dir)
	print(f"Adaptador de política PPO y métricas guardados en {output_dir}")


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"stage",
		choices=("reward", "ppo"),
		help="Entrenar reward model con preferencias o ajustar la política con PPO.",
	)
	parser.add_argument("--config", type=Path, default=project_root / "configs" / "default.yaml")
	parser.add_argument("--preference-file", type=Path, default=None)
	parser.add_argument("--policy-adapter", type=Path, default=None)
	parser.add_argument("--reward-adapter", type=Path, default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument(
		"--validate-only",
		action="store_true",
		help="Validar los CSV de anotación sin cargar modelos ni entrenar.",
	)
	args = parser.parse_args()
	config = load_config(args.config)
	phase_config = config["phase3"]
	preference_file = resolve_path(
		project_root,
		args.preference_file or phase_config["preference_file"],
	)
	rows = read_annotation_csv(preference_file, PREFERENCE_COLUMNS)
	validate_preferences(rows, config["phase1"]["tone_labels"])
	if args.validate_only:
		print(f"Comparaciones humanas válidas: {len(rows)}. Entrenamiento no ejecutado.")
		return

	if args.stage == "reward":
		output_dir = resolve_path(project_root, args.output_dir or phase_config["reward_model_dir"])
		train_reward_model(config, preference_file, output_dir)
	else:
		policy_adapter_dir = resolve_path(
			project_root,
			args.policy_adapter or phase_config["policy_adapter_dir"],
		)
		reward_adapter_dir = resolve_path(
			project_root,
			args.reward_adapter or phase_config["reward_model_dir"],
		)
		output_dir = resolve_path(project_root, args.output_dir or phase_config["output_dir"])
		train_ppo(config, preference_file, policy_adapter_dir, reward_adapter_dir, output_dir)


if __name__ == "__main__":
	main()