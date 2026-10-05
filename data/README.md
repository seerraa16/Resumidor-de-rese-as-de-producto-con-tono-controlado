# Anotaciones locales

Los datasets descargados, CSV de anotación y otros datos locales no se versionan.
Las plantillas vacías para preparar los dos tipos de entrenamiento están en
`data/templates/`.

## Ejemplos supervisados (Fase 2)

`sft_train.csv` requiere estas columnas:

- `review_text`: reseña original en inglés.
- `tone`: uno de los tonos configurados, inicialmente `neutral`, `friendly` o
  `formal`.
- `summary`: resumen escrito o revisado por una persona que respeta la reseña y
  el tono indicado.

Cada ejemplo enseña al modelo qué respuesta debería producir para ese prompt.
No uses automáticamente `review_title` como resumen objetivo.

`sft_eval.csv` tiene el mismo formato, pero debe contener reseñas separadas de
las filas de entrenamiento para evaluar generalización.

## Preferencias (Fase 3)

`preferences.csv` requiere estas columnas:

- `review_text`: la reseña original.
- `tone`: el tono pedido.
- `chosen`: el resumen que una persona prefiere para esa reseña y ese tono.
- `rejected`: una alternativa menos adecuada para exactamente el mismo prompt.

Una fila es una comparación, no una puntuación automática. Si generás las dos
alternativas con el modelo, una persona debe elegir cuál es mejor antes de usar
el par para entrenar. El reward model aprende las preferencias representadas
por esas decisiones; no descubre por sí solo qué tono es correcto.

Las plantillas solo contienen encabezados y no sirven para entrenar hasta que
se agreguen ejemplos reales. Usa comillas dobles alrededor de campos que
contengan comas o saltos de línea para mantener un CSV válido.
