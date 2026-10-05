"""Compare FLAN-T5 summaries requested in different tones."""

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data import load_reviews


TONE_INSTRUCTIONS = {
	"neutral": "Use a neutral, factual tone.",
	"friendly": "Use a warm, friendly tone while staying professional.",
	"formal": "Use a formal, polished tone.",
}


def build_prompt(review_text: str, tone: str) -> str:
	"""Build the same summarization instruction with a tone-specific request."""
	tone_instruction = TONE_INSTRUCTIONS.get(tone, f"Use a {tone} tone.")
	return (
		"Summarize the following product review in 1-2 concise sentences. "
		"Preserve the key experience and product details. "
		f"{tone_instruction}\n\nReview:\n{review_text}\n\nSummary:"
	)


def load_config(config_path: Path) -> dict[str, Any]:
	with config_path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	if not isinstance(config, dict):
		raise ValueError(f"La configuración no contiene un objeto YAML: {config_path}")
	return config


def generate_summary(
	prompt: str,
	tokenizer: Any,
	model: Any,
	device: torch.device,
	max_input_tokens: int,
	max_new_tokens: int,
) -> str:
	encoded_prompt = tokenizer(
		prompt,
		return_tensors="pt",
		truncation=True,
		max_length=max_input_tokens,
	).to(device)

	with torch.inference_mode():
		generated_tokens = model.generate(
			**encoded_prompt,
			max_new_tokens=max_new_tokens,
			num_beams=4,
			do_sample=False,
		)
	return tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()


def run_experiment(
	config: dict[str, Any],
	split: str,
	max_samples: int | None,
	output_path: Path,
	adapter_path: Path | None = None,
) -> int:
	data_config = config["data"]
	phase_config = config["phase1"]
	project_config = config["project"]
	sample_limit = max_samples if max_samples is not None else phase_config.get("max_samples")
	reviews = load_reviews(
		dataset_name=data_config["dataset_name"],
		dataset_config=data_config.get("dataset_config"),
		split=split,
		text_column=data_config["text_column"],
		reference_column=data_config.get("reference_column"),
		rating_column=data_config.get("rating_column"),
		max_samples=sample_limit,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_options = {"torch_dtype": torch.float16} if device.type == "cuda" else {}
	model_name = project_config["model_name"]
	print(f"Cargando {model_name} en {device}; puede descargarse la primera vez.")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_options).to(device)
	if adapter_path:
		from peft import PeftModel

		model = PeftModel.from_pretrained(model, str(adapter_path)).to(device)
	model.eval()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fields = [
		"sample_index",
		"tone",
		"rating",
		"reference_title",
		"review_text",
		"generated_summary",
	]
	output_count = 0
	with output_path.open("w", encoding="utf-8", newline="") as output_file:
		writer = csv.DictWriter(output_file, fieldnames=fields)
		writer.writeheader()
		for review in reviews:
			for tone in phase_config["tone_labels"]:
				prompt = build_prompt(review["review_text"], tone)
				summary = generate_summary(
					prompt=prompt,
					tokenizer=tokenizer,
					model=model,
					device=device,
					max_input_tokens=phase_config["max_input_tokens"],
					max_new_tokens=phase_config["max_new_tokens"],
				)
				writer.writerow(
					{
						"sample_index": review["sample_index"],
						"tone": tone,
						"rating": review.get("rating", ""),
						"reference_title": review.get("reference_title", ""),
						"review_text": review["review_text"],
						"generated_summary": summary,
					}
				)
				output_count += 1

	print(f"Guardadas {output_count} generaciones en {output_path}.")
	return output_count


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--config",
		type=Path,
		default=project_root / "configs" / "default.yaml",
		help="Ruta al archivo YAML de configuración.",
	)
	parser.add_argument("--split", default="test", help="Partición del dataset (por defecto: test).")
	parser.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Límite de reseñas; prevalece sobre la configuración.",
	)
	parser.add_argument("--output", type=Path, default=None, help="Ruta al CSV de salida.")
	parser.add_argument(
		"--adapter",
		type=Path,
		default=None,
		help="Adaptador LoRA entrenado en Fase 2 o afinado en Fase 3.",
	)
	args = parser.parse_args()

	config = load_config(args.config)
	output_path = args.output or Path(config["phase1"]["output_path"])
	run_experiment(config, args.split, args.max_samples, output_path, args.adapter)


if __name__ == "__main__":
	main()
