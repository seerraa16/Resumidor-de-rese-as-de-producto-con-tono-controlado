import csv
import tempfile
import unittest
from pathlib import Path

from src.data import prepare_reviews, read_annotation_csv
from src.phase2_lora import SFT_COLUMNS, build_dataset
from src.phase3_ppo import (
	PREFERENCE_COLUMNS,
	build_reward_dataset,
	validate_preferences,
)


class FakeTokenizer:
	def __call__(self, text=None, text_target=None, max_length=None, truncation=False):
		content = text_target if text_target is not None else text
		is_batch = isinstance(content, list)
		texts = content if is_batch else [content]
		input_ids = []
		for value in texts:
			tokens = list(range(1, len(value.split()) + 1))
			if truncation and max_length is not None:
				tokens = tokens[:max_length]
			input_ids.append(tokens or [1])
		result = {
			"input_ids": input_ids,
			"attention_mask": [[1] * len(tokens) for tokens in input_ids],
		}
		if text_target is not None:
			return result
		if not is_batch:
			return {key: value[0] for key, value in result.items()}
		return result


class TrainingDataTests(unittest.TestCase):
	def test_prepare_reviews_trims_text_and_skips_empty_rows(self):
		reviews = list(
			prepare_reviews(
				[
					{"body": " A useful product ", "title": "Useful", "stars": 5},
					{"body": "  ", "title": "Empty", "stars": 1},
				],
				text_column="body",
				reference_column="title",
				rating_column="stars",
			)
		)
		self.assertEqual(len(reviews), 1)
		self.assertEqual(reviews[0]["review_text"], "A useful product")
		self.assertEqual(reviews[0]["reference_title"], "Useful")
		self.assertEqual(reviews[0]["rating"], 5)

	def test_annotation_reader_rejects_header_only_csv(self):
		with tempfile.TemporaryDirectory() as temporary_directory:
			csv_path = Path(temporary_directory) / "empty.csv"
			with csv_path.open("w", encoding="utf-8", newline="") as output_file:
				csv.writer(output_file).writerow(SFT_COLUMNS)
			with self.assertRaisesRegex(ValueError, "no contiene ejemplos"):
				read_annotation_csv(csv_path, SFT_COLUMNS)

	def test_sft_builder_produces_encoder_decoder_fields(self):
		rows = [{"review_text": "Works well", "tone": "neutral", "summary": "A useful item."}]
		dataset = build_dataset(rows, FakeTokenizer(), ["neutral"], 32, 16)
		self.assertEqual(set(dataset.column_names), {"input_ids", "attention_mask", "labels"})
		self.assertEqual(len(dataset), 1)

	def test_reward_builder_produces_chosen_and_rejected_fields(self):
		rows = [
			{
				"review_text": "Works well",
				"tone": "neutral",
				"chosen": "A useful item.",
				"rejected": "It is okay.",
			}
		]
		dataset = build_reward_dataset(rows, FakeTokenizer(), ["neutral"], 48)
		self.assertEqual(
			set(dataset.column_names),
			{
				"input_ids_chosen",
				"attention_mask_chosen",
				"input_ids_rejected",
				"attention_mask_rejected",
			},
		)

	def test_preferences_reject_identical_answers(self):
		rows = [
			{"review_text": "Review", "tone": "neutral", "chosen": "Same", "rejected": "Same"}
		]
		with self.assertRaisesRegex(ValueError, "son iguales"):
			validate_preferences(rows, ["neutral"])


if __name__ == "__main__":
	unittest.main()