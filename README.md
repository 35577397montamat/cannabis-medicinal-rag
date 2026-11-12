# Cannabis Medicinal y Derechos — Asistente RAG para pacientes en Argentina

> **Estado del proyecto (12/11/2026):** prototipo académico funcional desarrollado con cinco notebooks de Jupyter y una aplicación independiente `app.py`. Ejecuté `app.py` desde la terminal de VS Code, confirmé la conexión con ChromaDB y sus 778 fragmentos, y abrí la interfaz Gradio en el navegador. 

## 1. Problema, objetivo y destinatarios

Desarrollé un asistente de **generación aumentada por recuperación (RAG)** para facilitar la consulta de información documental sobre cannabis medicinal, REPROCANN y derechos de los pacientes en Argentina. La normativa y los materiales institucionales se encuentran distribuidos entre distintas fuentes, por lo que reuní un corpus acotado y construí un sistema capaz de recuperar fragmentos y generar respuestas con referencias a los documentos consultados.

El asistente está dirigido a pacientes y personas interesadas en conocer información general sobre historia clínica, consentimiento informado, protección de datos personales, Ley 27.350, REPROCANN y ciertos derechos procesales, siempre dentro del alcance de los documentos recuperados.

**Alcance y límites:** el sistema no consulta trámites individuales de REPROCANN, no accede a bases de pacientes, no conoce estadísticas ausentes del corpus y no verifica automáticamente la vigencia normativa. Tampoco reemplaza el asesoramiento médico o jurídico. Las respuestas y las citas legales deben contrastarse con las fuentes oficiales vigentes.

## 2. Corpus documental y procedencia

Construí el corpus a partir de **nueve documentos públicos** vinculados con cannabis medicinal, REPROCANN, derechos de pacientes, datos personales y procedimientos judiciales. Consulté sitios oficiales e institucionales y guardé las páginas en PDF mediante la opción **«Imprimir → Guardar como PDF»** del navegador. Organicé los archivos en la carpeta `datos/`.

En el procesamiento inicial extraje texto de **268 páginas** y generé **778 fragmentos**. Conservé metadatos de archivo y página para identificar la procedencia de los pasajes recuperados.

| Documento utilizado | Fuente de referencia | Organismo o autoría identificada |
|---|---|---|
| `ATAJO_Dip Allanamientos-digital-a 150.pdf` | [Material sobre allanamientos](https://www.fiscales.gob.ar/wp-content/uploads/2015/12/Allanamientos.pdf) | Ministerio Público Fiscal, área de acceso a la justicia |
| `BOLETIN OFICIAL REPUBLICA ARGENTINA - MINISTERIO DE SALUD - Resolución 1780_2025.pdf` | [Boletín Oficial, Resolución 1780/2025](https://www.boletinoficial.gob.ar/detalleAviso/primera/325794/20250523) | Ministerio de Salud de la Nación |
| `Derechos de la persona imputada _ Unidad Especial Estratégica para la Implementación del Sistema Procesal Penal Acusatorio.pdf` | [Derechos de la persona imputada](https://www.mpf.gob.ar/unisa/derechos-de-la-persona-imputada/) | Ministerio Público Fiscal |
| `ley_26529_derechos_paciente.pdf` | [Ley 26.529, InfoLEG](https://servicios.infoleg.gob.ar/infolegInternet/anexos/160000-164999/160432/norma.htm) | Información Legislativa, Argentina |
| `PROTECCION DE LOS DATOS.pdf` | [Ley 25.326, InfoLEG](https://servicios.infoleg.gob.ar/infolegInternet/anexos/60000-64999/64790/texact.htm) | Información Legislativa, Argentina |
| `TEXTO ACTUALIZADO - Ley 23984 - CODIGO PROCESAL PENAL _ Argentina.gob.ar.pdf` | [Ley 23.984, texto actualizado](https://www.argentina.gob.ar/normativa/nacional/norma-383/actualizacion) | Congreso de la Nación / portal oficial |
| `TEXTO ACTUALIZADO - Ley 27350 - INVESTIGACION MEDICA Y CIENTIFICA _ Argentina.gob.ar.pdf` | [Cannabis medicinal, Argentina.gob.ar](https://www.argentina.gob.ar/justicia/derechofacil/leysimple/cannabis-medicinal) | Congreso de la Nación / portal oficial |
| `TEXTO ACTUALIZADO - Resolución 800_2021 - MINISTERIO DE SALUD _ Argentina.gob.ar.pdf` | [Resolución 800/2021, texto actualizado](https://www.argentina.gob.ar/normativa/nacional/norma-347837/actualizacion) | Ministerio de Salud de la Nación |
| `TEXTO ORIGINAL - Resolución 275_2016 - MINISTERIO DE SEGURIDAD _ Argentina.gob.ar.pdf` | [Resolución 275/2016, texto original](https://www.argentina.gob.ar/normativa/nacional/norma-263059/texto) | Ministerio de Seguridad de la Nación |

**Nota sobre procedencia y licencias:** estas referencias provienen de mi registro de fuenTES. La disponibilidad pública no equivale necesariamente a permiso de redistribución. Reconozco la autoría de los organismos originales y utilicé los documentos con fines académicos.

## 3. Arquitectura y herramientas

```text
Nueve PDF en datos/
    ↓ extracción de texto y metadatos
268 páginas
    ↓ RecursiveCharacterTextSplitter
      chunk_size=1000 / chunk_overlap=200
778 fragmentos → fragmentos.json
    ↓ OllamaEmbeddings: bge-m3:latest
ChromaDB persistente: ./chroma_db
    ↓ búsqueda por similitud, k=4 en la interfaz
Fragmentos recuperados + documento y página
    ↓ prompt restringido al contexto
OllamaLLM: granite4:latest (temperature=0)
    ↓
Respuesta y listado de fuentes en Gradio
```

Utilicé **Python 3.12**, **uv**, **VS Code**, **Jupyter**, **LangChain**, **ChromaDB**, **Ollama** y **Gradio** (versión 6.31.0 registrada durante el desarrollo). Los modelos fueron `bge-m3:latest` para embeddings y `granite4:latest` para generación. La colección de ChromaDB se llama `cannabis_derechos_bge_m3`.

También instalé **Rainbow CSV**, una extensión de VS Code que facilita la lectura visual de archivos CSV mediante el coloreado de columnas. Es una herramienta complementaria de edición y **no es una dependencia de Python ni interviene en el RAG**.

Los nombres `:latest` no garantizan que otra computadora descargue exactamente la misma versión de los modelos. Para una reproducción estricta convendría registrar sus identificadores y las versiones de las dependencias.

## 4. Organización de los notebooks

| Notebook | Trabajo realizado |
|---|---|
| `01_preparar_documentos.ipynb` | Cargué los PDF, extraje texto y metadatos, dividí el contenido en fragmentos y generé el archivo intermedio. |
| `02_base_vectorial.ipynb` | Generé embeddings con BGE-M3 y construí la colección persistente de ChromaDB. |
| `03_responder_con_fuentes.ipynb` | Implementé la generación de respuestas con Granite4, referencias a documentos, mejoras del prompt y auditoría manual de respuestas. |
| `04_evaluar_recuperacion.ipynb` | Medí la recuperación documental con nueve preguntas y distintas métricas y valores de `k`. |
| `05_interfaz_gradio.ipynb` | Implementé una interfaz local para ingresar consultas, visualizar respuestas y consultar los documentos recuperados. |

## 5. Instalación y ejecución local

Desarrollé el proyecto en Windows con Python 3.12. Para ejecutarlo en otro equipo se necesitan Python, `uv`, Ollama, las dependencias de Python, los modelos y el corpus documental correspondiente.

Desde la carpeta raíz del proyecto, en PowerShell:

```powershell
uv venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
ollama pull granite4:latest
ollama pull bge-m3:latest
ollama list
```

En VS Code, instalé las extensiones **Python** y **Jupyter** y seleccioné como kernel el intérprete `.venv\Scripts\python.exe`. Ollama debe estar disponible antes de realizar consultas.

**Interfaz:** primero ejecuté Gradio desde `05_interfaz_gradio.ipynb` con `interfaz.launch(inbrowser=True)`. Después generé `app.py` a partir del código del notebook y lo ejecuté desde la terminal de VS Code, en la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe app.py
```

Con el entorno virtual activado, también puedo usar `python app.py`. **Comprobé que la aplicación conecta con ChromaDB, encuentra 778 fragmentos y abre la interfaz Gradio en el navegador.** El sistema utiliza `granite4:latest` para generar respuestas y `bge-m3:latest` para los embeddings. Para realizar consultas, Ollama debe estar funcionando y la carpeta `chroma_db/` debe contener el índice construido previamente.

Esta prueba **valida el arranque local de `app.py`**.

**Reproducibilidad:** si el índice vectorial no está disponible, es necesario reconstruirlo a partir de los PDF con los notebooks 01 y 02. 

## 6. Evaluación de recuperación documental

En `04_evaluar_recuperacion.ipynb` utilicé **nueve preguntas** con un documento esperado por pregunta. Comparé la presencia de ese documento entre los primeros resultados recuperados.

| Métrica | Resultado |
|---|---:|
| Hit Rate@3 | 8/9 (88,89 %) |
| Hit Rate@5 | 8/9 (88,89 %) |
| Hit Rate@10 | 9/9 (100 %) |
| MRR@10 | 0,796 |
| Precision@3 aproximada por documento esperado | 0,667 (66,67 %) |

Observé que una consulta sobre la Resolución 800/2021 recuperó primero fragmentos de la Resolución 1780/2025 y que el documento esperado apareció en la **posición 6**.

Estas métricas miden **recuperación documental**, no exactitud jurídica ni calidad final de la respuesta. En particular, Precision@3 es una aproximación basada en la pertenencia al documento esperado; no evalué manualmente la relevancia individual de todos los fragmentos.

## 7. Auditoría de respuestas y mejoras del prompt

En el notebook 03 realicé una **auditoría manual de 15 preguntas** sobre el comportamiento del asistente. Según los criterios utilizados en esa revisión, clasifiqué los resultados así:

| Resultado de la auditoría | Casos |
|---|---:|
| Respuestas consideradas correctas | 7 |
| Abstenciones justificadas | 4 |
| Respuestas problemáticas pese a existir información relevante | 2 |
| Respuestas con información inventada o extrapolada | 2 |
| **Total** | **15** |

Consideré **11 de 15 casos (73,3 %) adecuados** bajo esa clasificación manual. Este porcentaje **no es una métrica formal de exactitud del modelo** y no debe extrapolarse a consultas nuevas.

### Errores y limitaciones observados

- **Acceso a datos del REPROCANN:** ante una pregunta sobre quién puede acceder a información de una persona inscripta, el sistema confundió los roles del registro con permisos de acceso a datos personales. Los fragmentos recuperados no respaldaban esa conclusión.
- **Viajes en avión con aceite de cannabis:** el sistema trasladó indebidamente a un pasajero particular un requisito de **Carta de Porte TAD** que el fragmento recuperado describía para asociaciones, fundaciones o personas jurídicas.
- **Derechos vinculados a la historia clínica:** en una prueba previa se observaron atribuciones incorrectas entre artículos y referencias normativas.
- **Modalidades de cultivo:** ante una consulta sobre modalidades contempladas por REPROCANN, el modelo confundió modalidades regulatorias con categorías como interior/exterior o produjo abstenciones inadecuadas.

### Ajustes y decisiones

Mejoré el prompt para exigir respuestas basadas en el contexto, citas de documento y página, abstención ante falta real de evidencia, tratamiento prudente de artículos legales y prohibición de inventar modalidades, categorías o requisitos. También incorporé la posibilidad de **responder parcialmente** cuando solo una parte de la pregunta está respaldada, y de utilizar enumeraciones explícitas cuando los documentos permiten contestar.

Probé comparativamente valores de **`k=4`, `k=6` y `k=8`** sobre cuatro consultas problemáticas. No observé una mejora confiable y uniforme al aumentar `k`; en algunos casos aparecieron nuevos errores. Por ese motivo **mantuve `k=4`** para la versión presentada. Al correr el notebook 03 creará un CSV de comparativa de respuestas con distintos números de K.

Posteriormente trasladé a Gradio las **reglas del prompt mejorado del notebook 03**, con el objetivo de mantener criterios consistentes entre la evaluación y la interfaz. No realicé una nueva evaluación cuantitativa completa después de esa actualización.

## 8. Interfaz Gradio y pruebas funcionales

Implementé una interfaz local con un campo para escribir preguntas, un botón para consultar, una opción para limpiar y preguntas de ejemplo. La aplicación muestra la respuesta generada y los documentos recuperados, con información de procedencia.

**Estado verificado:** abrí la interfaz desde el notebook 05 y, posteriormente, ejecuté `app.py` desde la terminal. En esta última ejecución confirmé la conexión a ChromaDB, el recuento de 778 fragmentos y la apertura de Gradio. Las pruebas funcionales registradas anteriormente incluyen consultas respondibles y consultas para las que el sistema reconoció falta de información; todavía no documenté una nueva batería completa de respuestas de `app.py`.

| Consulta de prueba | Observación registrada |
|---|---|
| Titularidad de la historia clínica y plazo de entrega de copia según Ley 26.529 | Se obtuvo una respuesta pertinente en una prueba anterior, aunque no siempre se citó la página junto a cada afirmación. |
| Objetivo de la Ley 27.350 | Se obtuvo una respuesta pertinente en el notebook 03. |
| Protección de datos personales según Ley 25.326 | Se obtuvo una respuesta basada en el documento correspondiente en el notebook 03. |
| Cantidad de inscriptos a REPROCANN durante octubre de 2026 | En una prueba del notebook 05 se indicó falta de información suficiente. |
| Estado individual de un trámite REPROCANN | En una prueba del notebook 05 se indicó que no era posible determinarlo. |


## 9. Limitaciones técnicas, jurídicas y de reproducibilidad

- **Normativa:** el corpus refleja documentos seleccionados en un momento determinado; el asistente no comprueba modificaciones posteriores ni vigencia actual.
- **Citas:** el listado de fuentes muestra fragmentos recuperados, pero no verifica automáticamente que cada afirmación generada esté respaldada por una cita exacta.
- **Fragmentación:** dividir textos jurídicos en fragmentos de 1000 caracteres con solapamiento de 200 puede separar artículos de sus encabezados y favorecer confusiones.
- **Recuperación:** aumentar `k` puede aportar cobertura, pero también introducir fragmentos que distraigan al modelo.
- **Privacidad:** no incorporé datos de pacientes ni mecanismos de consulta a expedientes o trámites individuales.
- **Entorno Windows:** al reabrir el notebook 01 observé un error de importación de `_uuid_utils` con el mensaje «Una directiva de Control de aplicaciones bloqueó este archivo». En una ejecución posterior de la celda de fragmentación, la importación avanzó, pero apareció `NameError: name 'documentos' is not defined` porque la variable no había sido cargada en esa sesión. **No realicé una reejecución completa del notebook 01 ni una instalación limpia para resolver o descartar definitivamente el bloqueo.** Estos incidentes no demuestran pérdida de los 778 fragmentos ni de la colección existente, pero impiden garantizar la reproducción sin pruebas adicionales.

## 10. Conservación del índice y precauciones al reejecutar

El archivo `app.py` consulta el índice existente, sin volver a fragmentar los PDF ni reconstruir embeddings. La carpeta `chroma_db/` contiene el índice persistente y `fragmentos.json` es un archivo intermedio. Para preservar el trabajo realizado, decidí **no volver a ejecutar íntegramente los notebooks 01 y 02** y si bien se observa un error (se debe a que no esta ejecutado completo nuevamente).

## 11. Uso de herramientas de inteligencia artificial

Utilicé **ChatGPT (OpenAI)** como apoyo durante la planificación, elaboración de código, la comprensión de bibliotecas de Python, la depuración, la interpretación de métricas, las pruebas y la redacción de la documentación. Utilicé **Granite4 mediante Ollama** para generar las respuestas y **BGE-M3 mediante Ollama** para producir embeddings.

Ejecuté y revisé las etapas del proyecto en mi equipo. Documenté tanto los resultados favorables como las abstenciones, errores de atribución normativa y limitaciones detectadas. La auditoria exhaustiva de todas las respuestas y la verificación definitiva de procedencia y licencias del corpus permanecen pendientes.


**Conclusión:** logré implementar un prototipo RAG funcional, con corpus institucional, recuperación vectorial, generación local, evaluación de recuperación y una interfaz de consulta que también ejecuté de forma independiente mediante `app.py`. La auditoría mostró que la recuperación de documentos puede ser satisfactoria sin que eso garantice respuestas jurídicas correctas. Por eso, presento el sistema como **herramienta informativa experimental**, con límites explícitos y necesidad de verificación humana de las fuentes.
