"""Load and normalize review records from a Hugging Face dataset."""

from collections.abc import Iterable, Iterator, Mapping
import csv
from pathlib import Path
from typing import Any


def prepare_reviews(
	rows: Iterable[Mapping[str, Any]],
	text_column: str,
	reference_column: str | None = None,
	rating_column: str | None = None,
) -> Iterator[dict[str, Any]]:
	"""Normalize rows without treating review titles as gold summaries."""
	for sample_index, row in enumerate(rows):
		review_text = row.get(text_column)
		if not isinstance(review_text, str) or not review_text.strip():
			continue

		review = {
			"sample_index": sample_index,
			"review_text": review_text.strip(),
		}
		if reference_column:
			reference = row.get(reference_column)
			review["reference_title"] = str(reference).strip() if reference is not None else ""
		if rating_column and rating_column in row:
			review["rating"] = row[rating_column]

		yield review


def load_reviews(
	dataset_name: str,
	dataset_config: str | None,
	split: str,
	text_column: str,
	reference_column: str | None = None,
	rating_column: str | None = None,
	max_samples: int | None = None,
) -> Iterator[dict[str, Any]]:
	"""Load a dataset split, optionally limiting it to its first N rows."""
	if max_samples is not None and max_samples <= 0:
		raise ValueError("max_samples debe ser mayor que cero.")

	try:
		from datasets import load_dataset
	except ImportError as error:
		raise RuntimeError(
			"Falta la dependencia 'datasets'. Instalá requirements.txt en el entorno activo."
		) from error

	split_spec = f"{split}[:{max_samples}]" if max_samples is not None else split
	if dataset_config:
		dataset = load_dataset(dataset_name, dataset_config, split=split_spec)
	else:
		dataset = load_dataset(dataset_name, split=split_spec)

	required_columns = [text_column]
	if reference_column:
		required_columns.append(reference_column)
	missing_columns = [column for column in required_columns if column not in dataset.column_names]
	if missing_columns:
		available = ", ".join(dataset.column_names)
		raise ValueError(
			f"Columnas no encontradas en {dataset_name}: {', '.join(missing_columns)}. "
			f"Columnas disponibles: {available}. Revisá la configuración."
		)

	return prepare_reviews(
		dataset,
		text_column=text_column,
		reference_column=reference_column,
		rating_column=rating_column,
	)


def read_annotation_csv(file_path: Path, required_columns: tuple[str, ...]) -> list[dict[str, str]]:
	"""Read a non-empty annotation CSV and report missing fields clearly."""
	if not file_path.is_file():
		raise FileNotFoundError(
			f"No existe {file_path}. Copiá la plantilla correspondiente de data/templates/ "
			"a data/annotations/ y completala con anotaciones humanas."
		)

	with file_path.open(encoding="utf-8-sig", newline="") as input_file:
		reader = csv.DictReader(input_file)
		if reader.fieldnames is None:
			raise ValueError(f"El CSV no tiene cabecera: {file_path}")
		missing_columns = [column for column in required_columns if column not in reader.fieldnames]
		if missing_columns:
			raise ValueError(
				f"Faltan columnas en {file_path}: {', '.join(missing_columns)}. "
				f"Columnas requeridas: {', '.join(required_columns)}."
			)

		rows = []
		for row_number, row in enumerate(reader, start=2):
			clean_row = {column: (row.get(column) or "").strip() for column in required_columns}
			empty_columns = [column for column, value in clean_row.items() if not value]
			if empty_columns:
				raise ValueError(
					f"Hay campos vacíos en la fila {row_number} de {file_path}: "
					f"{', '.join(empty_columns)}."
				)
			rows.append(clean_row)

	if not rows:
		raise ValueError(
		f"{file_path} no contiene ejemplos anotados. No se puede entrenar con solo la cabecera."
	)
	return rows
