# Resumidor de reseñas con tono controlado

Proyecto personal de NLP para generar resúmenes de reseñas de producto con un
tono elegible. El modelo inicial es
[`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base). La carga de
datos, la Fase 1, el entrenamiento supervisado LoRA y el flujo de reward model +
PPO están implementados. El entrenamiento requiere CSV con anotaciones humanas;
no se inventan resúmenes ni preferencias a partir de títulos o estrellas.

## Fases

1. **Prompt engineering:** comparar resúmenes con instrucciones de tono neutral,
   friendly y formal.
2. **Fine-tuning eficiente:** LoRA/PEFT aprende de ejemplos `reseña + tono → resumen`.
3. **Alineamiento:** un reward model aprende de pares humanos `elegido/rechazado`;
    después PPO ajusta el adaptador de Fase 2 usando esa recompensa.

## Datos

La configuración provisional usa `amazon_reviews_multi` en inglés. Se cargan
reseñas de `review_body`; `review_title` se conserva como metadato y `stars`
como valoración. El título no se considera un resumen de referencia y no se usa
para calcular ROUGE. Como alternativa se puede configurar `yelp_review_full`
con las columnas `text` y `label` (sin columna de título).

La carga ocurre al ejecutar Fase 1. La primera ejecución puede descargar datos y
el modelo desde Hugging Face. `amazon_reviews_multi` aporta texto, título y
valoración, pero no resúmenes de referencia ni preferencias; por eso Fase 2 y
Fase 3 requieren anotaciones humanas propias. Las plantillas vacías están en
`data/templates/`; los archivos de trabajo van en `data/annotations/`, ignorado
por Git. Revisá licencias y condiciones del dataset antes de publicar.

## Requisitos e instalación

Se recomienda Python 3.10, 3.11 o 3.12, Git y una GPU NVIDIA con controlador
compatible. Este entorno usa Python 3.12.7. La RTX 4060 de 8 GB sirve para FLAN-T5 base con los lotes pequeños
configurados; aun así, los entrenamientos pueden tardar y conviene empezar con
pocos ejemplos. PPO mantiene además el modelo de recompensa y el de política en
memoria, así que hay que vigilar el uso de VRAM.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux o macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Instalá PyTorch **por separado** desde el selector oficial de
[pytorch.org](https://pytorch.org/get-started/locally/). Elegí sistema, `Pip`,
`Python` y una versión CUDA compatible con tu controlador; ejecutá dentro del
entorno virtual el comando que genere el sitio. No uses `pip install torch` a
secas si necesitás CUDA, porque podría instalar una variante sin el soporte que
buscás.

Después instalá las demás dependencias y comprobá la GPU:

```bash
python -m pip install -r requirements.txt
python src/check_gpu.py
```

Las versiones del stack Hugging Face están fijadas en `requirements.txt` porque
el PPO clásico usado aquí corresponde a TRL 0.11.4.

## Fase 1: prompt engineering

Desde la raíz del repositorio, ejecutá:

```bash
python -m src.phase1_prompting --max-samples 2
```

Por defecto se toman hasta 10 ejemplos de `test`. Para cada reseña, el mismo
modelo genera una salida por cada tono, con búsqueda determinista. El programa
guarda una fila por resultado en `results/phase1_outputs.csv`. La comparación es
cualitativa: revisá fidelidad, claridad, concisión y tono. No se calcula ROUGE
porque el dataset elegido no aporta resúmenes de referencia adecuados.

Se pueden cambiar el split, límite de ejemplos, modelo, tonos y longitudes en
`configs/default.yaml`; también están disponibles `--split`, `--max-samples`,
`--output` y `--config` en la línea de comandos.

## Preparar anotaciones

Copiá las plantillas sin modificar las cabeceras:

```powershell
New-Item -ItemType Directory -Force data/annotations
Copy-Item data/templates/sft_train.csv data/annotations/sft_train.csv
Copy-Item data/templates/sft_eval.csv data/annotations/sft_eval.csv
Copy-Item data/templates/preferences.csv data/annotations/preferences.csv
```

En `sft_train.csv`, cada fila debe tener una reseña real, un tono de la lista
configurada y un resumen escrito/revisado por una persona. Reservá ejemplos
distintos en `sft_eval.csv` para evaluación. En `preferences.csv`, cada fila
contiene una reseña, su tono, el resumen elegido y otro rechazado para el mismo
prompt. Las cabeceras están explicadas en `data/README.md`. El código rechaza
campos vacíos, tonos desconocidos y pares idénticos; no entrenes con las
plantillas vacías.

## Fase 2: LoRA supervisado

Primero validá los CSV sin cargar el modelo:

```powershell
python -m src.phase2_lora --validate-only
```

Con anotaciones completas, iniciá el ajuste:

```powershell
python -m src.phase2_lora
```

El script transforma cada reseña y tono en un prompt, usa el resumen humano como
objetivo y aprende adaptadores LoRA sobre FLAN-T5. Por defecto guarda el
adaptador en `results/phase2_lora/`. No modifica ni vuelve a guardar el modelo
base. Con el CSV `sft_eval.csv` configurado, calcula ROUGE-1, ROUGE-2 y ROUGE-L
contra los resúmenes humanos y guarda métricas de entrenamiento/evaluación en
el directorio del adaptador.

## Fase 3: reward model y PPO

Validá las comparaciones humanas y entrena primero el reward model:

```powershell
python -m src.phase3_ppo reward --validate-only
python -m src.phase3_ppo reward
```

Luego ajustá con PPO el adaptador producido en Fase 2:

```powershell
python -m src.phase3_ppo ppo
```

El reward model aprende a puntuar más alto el resumen elegido que el rechazado.
PPO genera resúmenes para los prompts anotados, obtiene esa puntuación y ajusta
la política con una penalización KL frente al adaptador de referencia. Los
hiperparámetros y límites cortos de prueba están en `configs/default.yaml`; los
resultados y métricas se guardan bajo `results/phase3_reward/` y
`results/phase3_ppo/`.

TRL 0.11.4 usa la API PPO clásica y de bajo nivel. Es intencional que aquí quede
fijada esa versión. Los valores iniciales son una configuración experimental,
no una garantía de alineamiento: revisá generaciones y reward scores, y no
interpretes una subida de reward como calidad real sin evaluación humana.

## Estructura

```text
configs/default.yaml       Configuración de datos y generación
notebooks/                 Exploración puntual de resultados
results/                   CSV y salidas locales ignoradas por Git
src/data.py                Carga y normalización de reseñas
src/phase1_prompting.py    Generación comparativa por tono
src/phase2_lora.py         Fine-tuning supervisado LoRA/PEFT
src/phase3_ppo.py          Reward model y alineamiento PPO
src/utils.py               Placeholder para utilidades compartidas
src/check_gpu.py           Diagnóstico PyTorch/CUDA
data/templates/            Cabeceras para anotación supervisada y preferencias
data/annotations/           Anotaciones locales, excluidas de Git
```

Los scripts entrenan solo cuando se ejecutan explícitamente. No subas
checkpoints, datasets, anotaciones, cachés ni tokens de Hugging Face al
repositorio.
