"""Supervised LoRA fine-tuning of FLAN-T5 on human-written summaries."""

import argparse
from pathlib import Path
from typing import Any

import torch
import numpy as np
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from rouge_score import rouge_scorer
from transformers import (
	AutoModelForSeq2SeqLM,
	AutoTokenizer,
	DataCollatorForSeq2Seq,
	Seq2SeqTrainer,
	Seq2SeqTrainingArguments,
)

from src.data import read_annotation_csv
from src.phase1_prompting import build_prompt, load_config


SFT_COLUMNS = ("review_text", "tone", "summary")


def build_compute_metrics(tokenizer: Any):
	"""Build mean ROUGE scores against human-written evaluation summaries."""
	scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

	def compute_metrics(eval_prediction: Any) -> dict[str, float]:
		predictions = eval_prediction.predictions
		if isinstance(predictions, tuple):
			predictions = predictions[0]
		labels = np.where(eval_prediction.label_ids == -100, tokenizer.pad_token_id, eval_prediction.label_ids)
		decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
		decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
		all_scores = [
			scorer.score(reference, prediction)
			for reference, prediction in zip(decoded_labels, decoded_predictions)
		]
		return {
			f"{metric}_fmeasure": float(np.mean([score[metric].fmeasure for score in all_scores]))
			for metric in ("rouge1", "rouge2", "rougeL")
		}

	return compute_metrics


def resolve_path(project_root: Path, value: str | Path) -> Path:
	path = Path(value)
	return path if path.is_absolute() else project_root / path


def build_dataset(
	rows: list[dict[str, str]],
	tokenizer: Any,
	tones: list[str],
	max_input_tokens: int,
	max_target_tokens: int,
) -> Dataset:
	"""Tokenize annotated input/target pairs for an encoder-decoder model."""
	invalid_tones = sorted({row["tone"] for row in rows} - set(tones))
	if invalid_tones:
		raise ValueError(f"Tonos no definidos en configs/default.yaml: {', '.join(invalid_tones)}")

	def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
		prompts = [build_prompt(text, tone) for text, tone in zip(batch["review_text"], batch["tone"])]
		encoded = tokenizer(prompts, max_length=max_input_tokens, truncation=True)
		labels = tokenizer(text_target=batch["summary"], max_length=max_target_tokens, truncation=True)
		encoded["labels"] = labels["input_ids"]
		return encoded

	return Dataset.from_list(rows).map(
		tokenize_batch,
		batched=True,
		remove_columns=list(SFT_COLUMNS),
	)


def train(config: dict[str, Any], train_file: Path, eval_file: Path | None, output_dir: Path) -> None:
	phase_config = config["phase2"]
	train_rows = read_annotation_csv(train_file, SFT_COLUMNS)
	eval_rows = read_annotation_csv(eval_file, SFT_COLUMNS) if eval_file else None
	print(f"Ejemplos supervisados de entrenamiento: {len(train_rows)}")
	if eval_rows:
		print(f"Ejemplos de evaluación: {len(eval_rows)}")

	model_name = config["project"]["model_name"]
	device_has_cuda = torch.cuda.is_available()
	dtype = torch.float16 if device_has_cuda else torch.float32
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
	model.config.use_cache = False

	lora_config = LoraConfig(
		task_type=TaskType.SEQ_2_SEQ_LM,
		r=phase_config["lora_r"],
		lora_alpha=phase_config["lora_alpha"],
		lora_dropout=phase_config["lora_dropout"],
		target_modules=phase_config["target_modules"],
		bias="none",
	)
	model = get_peft_model(model, lora_config)
	model.print_trainable_parameters()
	train_dataset = build_dataset(
		train_rows,
		tokenizer,
		config["phase1"]["tone_labels"],
		phase_config["max_input_tokens"],
		phase_config["max_target_tokens"],
	)
	eval_dataset = (
		build_dataset(
			eval_rows,
			tokenizer,
			config["phase1"]["tone_labels"],
			phase_config["max_input_tokens"],
			phase_config["max_target_tokens"],
		)
		if eval_rows
		else None
	)

	training_args = Seq2SeqTrainingArguments(
		output_dir=str(output_dir),
		num_train_epochs=phase_config["num_train_epochs"],
		learning_rate=phase_config["learning_rate"],
		per_device_train_batch_size=phase_config["per_device_train_batch_size"],
		per_device_eval_batch_size=phase_config["per_device_eval_batch_size"],
		gradient_accumulation_steps=phase_config["gradient_accumulation_steps"],
		fp16=device_has_cuda,
		gradient_checkpointing=bool(phase_config.get("gradient_checkpointing", False)),
		remove_unused_columns=bool(phase_config.get("remove_unused_columns", False)),
		logging_steps=phase_config["logging_steps"],
		eval_strategy="epoch" if eval_dataset else "no",
		generation_max_length=phase_config["max_target_tokens"],
		save_strategy="epoch",
		save_total_limit=2,
		predict_with_generate=True,
		report_to="none",
		seed=config["training"]["seed"],
		data_seed=config["training"]["seed"],
	)
	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
		tokenizer=tokenizer,
		compute_metrics=build_compute_metrics(tokenizer) if eval_dataset else None,
	)
	train_result = trainer.train()
	trainer.save_model(str(output_dir))
	tokenizer.save_pretrained(output_dir)
	trainer.save_metrics("train", train_result.metrics)
	if eval_dataset:
		trainer.save_metrics("eval", trainer.evaluate())
	print(f"Adaptador LoRA guardado en {output_dir}")


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=project_root / "configs" / "default.yaml")
	parser.add_argument("--train-file", type=Path, default=None)
	parser.add_argument("--eval-file", type=Path, default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--validate-only", action="store_true", help="Validar anotaciones sin cargar el modelo.")
	args = parser.parse_args()
	config = load_config(args.config)
	phase_config = config["phase2"]
	train_file = resolve_path(project_root, args.train_file or phase_config["train_file"])
	eval_value = args.eval_file or phase_config.get("eval_file")
	eval_file = resolve_path(project_root, eval_value) if eval_value else None
	output_dir = resolve_path(project_root, args.output_dir or phase_config["output_dir"])

	train_rows = read_annotation_csv(train_file, SFT_COLUMNS)
	if eval_file:
		read_annotation_csv(eval_file, SFT_COLUMNS)
	if args.validate_only:
		print(f"Anotaciones válidas: {len(train_rows)}. Entrenamiento no ejecutado.")
		return
	train(config, train_file, eval_file, output_dir)


if __name__ == "__main__":
	main()