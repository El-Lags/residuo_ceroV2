# ♻️ Residuo Cero · clasificador de residuos con IA

App web que clasifica la foto de un residuo (cartón, vidrio, metal, papel, plástico o basura), avisa cuando la imagen no parece un residuo, muestra la confianza de cada clase y genera una recomendación de reciclaje con un modelo de lenguaje (Gemini).

**Proyecto grupal · Certamen · IA Generativa para la Gestión Industrial · UDD**

🔗 **App publicada:** https://residuocerov2.streamlit.app/
📦 **Repositorio:** https://github.com/El-Lags/residuo_ceroV2

## Integrantes
- Felipe Alonso
- Sofía Fariña
- Vicente Lagunas

## Flujo
📷 Imagen → 🧠 EfficientNet-B0 (clase + confianza) → 📊 Resultado visual → 💬 Gemini → ✅ Recomendación

## Problema
Separar mal los residuos contamina lotes completos de material reciclable. La app ayuda a identificar el material y orienta sobre cómo y dónde reciclarlo, con aviso de revisión humana cuando el modelo no está seguro.

## Datos
El modelo se entrena con **tres fuentes** y el test se mide **por fuente**, para no esconder cómo funciona fuera de TrashNet.

| Fuente | Qué aporta | Fotos usadas | Licencia |
|---|---|---|---|
| [TrashNet](https://huggingface.co/datasets/garythung/trashnet) (Thung y Yang, Stanford, 2016; copia publicada en Hugging Face por `garythung`), archivo `dataset-resized.zip` | 6 clases de residuos, un objeto sobre fondo liso | 2.524 (de 2.527; se quitaron 3 copias exactas) | MIT |
| [Fotos de internet](https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes) (12 carpetas, 15.520 fotos) | latas, botellas, cartón, papel, plástico, vidrio, basura, restos orgánicos, zapatos, ropa y pilas, con fondos y encuadres variados | 5.087 | no declarada; solo uso educativo |
| [ImageNet-1k](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256) (Deng et al., 2009; copia con las fotos reducidas a 256×256 publicada en Hugging Face por `benjamin-paine`; split de validación) | 50 categorías de objetos cotidianos para `other`: calzado, mochilas, electrónicos, muebles, animales, vehículos y paisajes | 250 | licencia «other»: términos propios de ImageNet (investigación y educación sin fines comerciales) |

- **Cómo se asignan las clases:** `glass` = vidrio marrón, verde y blanco; `trash` = basura y restos orgánicos (van a la basura general); `other` = zapatos, ropa, pilas (no van a un contenedor de reciclaje domiciliario) y los objetos de ImageNet. De ImageNet se excluyeron las categorías que se parecen a residuos (botellas, latas, cartones, bolsas, libros, tazas).
- **Duplicados entre fuentes:** el dataset de internet **incluye las fotos de TrashNet**, idénticas byte a byte (el 45 % de su carpeta de cartón, el 53 % de metal, el 57 % de papel y el 55 % de plástico). Se eliminan las copias exactas antes de dividir los datos; así el test de TrashNet no se filtra al entrenamiento.
- **Total: 7.861 fotos.** El desbalance es moderado (de 2,4 a 1 entre la clase mayor, `glass`, y la menor, `metal`).

| Clase | TrashNet | Internet | ImageNet | Total |
|---|---|---|---|---|
| cardboard | 403 | 488 | 0 | 891 |
| glass | 501 | 1.350 | 0 | 1.851 |
| metal | 409 | 359 | 0 | 768 |
| other | 0 | 1.050 | 250 | 1.300 |
| paper | 594 | 456 | 0 | 1.050 |
| plastic | 480 | 387 | 0 | 867 |
| trash | 137 | 997 | 0 | 1.134 |
| **Total** | **2.524** | **5.087** | **250** | **7.861** |

- **División estratificada por clase y por fuente (semilla 1993):**

| Clase | Train | Val | Test |
|---|---|---|---|
| cardboard | 623 | 134 | 134 |
| glass | 1.296 | 277 | 278 |
| metal | 537 | 116 | 115 |
| other | 910 | 195 | 195 |
| paper | 735 | 157 | 158 |
| plastic | 607 | 130 | 130 |
| trash | 794 | 170 | 170 |
| **Total** | **5.502** | **1.179** | **1.180** |

- **Reservado aparte** (no entra a train, val ni test): 30 zapatos y 40 fotos de 8 categorías de ImageNet que el modelo nunca ve (pizza, castillo, tractor, león, microondas, refrigerador, lavadora y aspiradora).
- **Limitaciones de los datos:** las fotos de internet tienen etiquetas ruidosas y muchas casi duplicadas (solo se eliminan copias exactas); `trash` agrupa cosas muy distintas (basura de TrashNet, basura de internet y restos orgánicos); `other` no cubre todos los objetos posibles (por ejemplo, madera no está); gran parte de las fotos no corresponde a envases chilenos.
- **Las pilas son un caso especial:** se agruparon en `other` porque no van a ningún contenedor domiciliario, pero sí son residuos (peligrosos) que deben llevarse a un punto de recolección especial. Por eso el mensaje «No parece un residuo» es impreciso para ellas.
- Los datasets **no** están en este repositorio por su tamaño; se descargan desde los enlaces.
- **Sobre las vistas previas en Hugging Face:** muestran solo las primeras filas, ordenadas por clase. En el dataset de internet las primeras son de la carpeta `battery` (pilas) y en ImageNet de la categoría 0 (un pez, la tenca), pero ambos contienen todas las clases que se listan arriba. De ImageNet solo se usaron 50 de las 1.000 categorías (lista en `CLASES_IN` del notebook).

## Modelo
| Modelo | Accuracy (test) | Accuracy balanceada (test) |
|---|---|---|
| CLIP zero-shot (línea base) | 58,4 % | 55,1 % |
| **EfficientNet-B0 (timm), transfer learning** | **93,0 %** | **93,1 %** |

Elegimos EfficientNet-B0 porque supera a CLIP por 34,6 puntos de accuracy y 38,0 de accuracy balanceada con las mismas 1.180 imágenes de prueba, y además pesa unos 16 MB, cabe en GitHub y corre en CPU. Es un modelo de **7 clases** (6 materiales y `other`).

| Clase | Recall (test) | en fotos de TrashNet | en fotos de internet |
|---|---|---|---|
| cardboard | 97,0 % | 93 % | 100 % |
| metal | 96,5 % | 97 % | 96 % |
| paper | 96,2 % | 96 % | 97 % |
| glass | 94,2 % | 91 % | 96 % |
| other | 89,7 % | — | 94 % (ImageNet: 73 %) |
| plastic | 89,2 % | 83 % | 97 % |
| trash | 88,8 % | 75 % (solo 20 fotos) | 91 % |

**Accuracy por fuente:** TrashNet 91,0 %, internet 94,9 %, ImageNet 73,0 %. Las fotos de internet tienen casi-duplicados entre sí, así que ese 94,9 % puede ser algo optimista.

**Error más costoso:** clasificar como reciclable un residuo no reciclable, porque contamina el lote completo. Ocurrió con 15 de las 170 imágenes `trash` del test (8,8 %): 5 como metal, 4 como cartón, 3 como plástico, 2 como vidrio y 1 como papel. El error inverso también cuesta, porque se pierde material reciclable: 5 de 815 imágenes reciclables se clasificaron como `trash` (2 vidrio, 2 plástico y 1 papel) y 2 como `other` (papel).

**Confianza:** la confianza media es 0,97 cuando el modelo acierta y 0,75 cuando se equivoca. El 94 % de las fotos tiene confianza de 0,7 o más y en ese grupo el accuracy es 95,7 %; el 4,1 % del test se clasifica mal con confianza de 0,7 o más, y en esos casos la app no avisa.

**Objetos que no son residuos (problema original):** antes, un zapato se clasificaba como vidrio con 92 % de confianza. Ahora los 30 zapatos reservados se detectan como `other` (100 %) y 25 de las 40 fotos de categorías nunca vistas (62,5 %); las 15 que fallan se van a `trash` (5), papel (4), metal (3), cartón (2) y plástico (1). Solo 6 de 985 residuos del test (0,6 %) se marcan como `other`.

### Pruebas independientes: fotos que el modelo nunca vio
Las cifras de arriba vienen de fotos de las mismas fuentes que el entrenamiento. Para saber cómo funciona con fotos realmente nuevas se hicieron tres pruebas más (scripts en `pruebas/`), comparando con el modelo anterior (entrenado solo con TrashNet y 400 fotos de `other`):

| Prueba | Modelo anterior | **Modelo actual** |
|---|---|---|
| 1.374 fotos de internet que el entrenamiento nunca usó (vidrio, zapatos, ropa, pilas, restos orgánicos) | 59,8 % | **94,6 %** |
| 77 fotos de residuos de [Wikimedia Commons](https://commons.wikimedia.org) (latas, botellas, vidrio, cartón, papel) | 57,1 % | **81,8 %** |
| 154 fotos de Wikimedia Commons de objetos que no son residuos (madera, zapatos, mochilas, laptops, sillas, gatos, bicicletas, flores) | 61,0 % | **72,1 %** |

- **Detalle de Wikimedia Commons (modelo actual):** latas 65 %, botellas de plástico 79 %, vidrio 80 %, cartón 94 %, papel 100 %, madera 64 %, zapatos 70 %, bicicletas 100 %, flores 35 %.
- **Errores sin aviso:** en las fotos de residuos de Commons, el modelo se equivoca con confianza de 70 % o más en el 14,3 % de los casos. Dice «No parece un residuo» a un residuo real en el 5,2 %.
- **Fotos probadas a mano:** una lata Heineken y una lata gris sin marca se reconocen como metal (99 % y 100 %); dos fotos de una lata Fanta salen como plástico y una foto de madera sale como metal con 43 % y papel 35 %.
- **Cómo se hizo:** las fotos de Commons (licencias abiertas, no incluidas en el repositorio) se eligieron a ojo, buscando que el objeto fuera el protagonista, **antes** de ver ninguna predicción. Son pocas fotos (77 de residuos), así que cada cifra tiene unos ±10 puntos de margen.
- **Una corrección importante:** el dataset de internet incluye las fotos de TrashNet. Una primera prueba externa con ese dataset, sin quitar esas copias, sobreestimaba el acierto del modelo anterior (47 % en lugar de 22 % en las fotos realmente nuevas).

**Estimación realista:** con fotos nuevas, el modelo acierta cerca de **80 % en residuos**, no el 93 % del test.

### Limitaciones conocidas
- Las **latas enteras** se confunden con vidrio o plástico (65 % de acierto en las fotos de Commons), a veces con confianza alta.
- **Plástico y vidrio** se confunden entre sí en botellas; las **flores y plantas** se clasifican como `trash` (porque `trash` incluye restos orgánicos).
- `other` se entrenó con pocas fotos de ImageNet (250), así que objetos como la madera o los muebles fallan con frecuencia.
- Los **PNG con fondo transparente** se leen con fondo negro, lo que puede cambiar el resultado.
- No se midió el rendimiento con envases chilenos.
- La confianza alta no garantiza que el resultado sea correcto: es un prototipo educativo.

## Estructura
```
app.py                       aplicación Streamlit
requirements.txt             librerías
notebooks/entrenamiento.ipynb  datos, comparación de modelos y métricas
modelo/                      pesos (.pt), clases y métricas del test
ejemplos/                    imágenes de prueba
pruebas/                     scripts de las pruebas independientes (Wikimedia Commons)
.streamlit/config.toml       tema de la app
```

## Cómo ejecutarla
```bash
pip install -r requirements.txt
# clave de Gemini (nunca se sube a GitHub)
mkdir .streamlit   # si no existe
echo GEMINI_API_KEY = "tu_clave" > .streamlit/secrets.toml
streamlit run app.py
```
En Streamlit Cloud la clave se guarda en **Advanced settings → Secrets** como `GEMINI_API_KEY = "..."`.

## Uso de IA

- **Modelos de visión:** Se utilizó EfficientNet-B0 (`timm`), ajustado con TrashNet, fotos de internet (dataset de 12 clases) y la clase `other` con imágenes de ImageNet. CLIP (OpenAI) se empleó como modelo de referencia.

- **IA generativa en la aplicación:** Google Gemini se utiliza para generar recomendaciones textuales a partir de la clase predicha y su nivel de confianza. La imagen no es enviada al modelo.

- **Asistente de programación (Claude Code, de Anthropic):** se usó durante el desarrollo como apoyo, no como reemplazo del trabajo del equipo. Concretamente ayudó con:
  - escribir y ordenar el código del notebook de entrenamiento y de `app.py` (interfaz, estilos, caché de recomendaciones por imagen);
  - preparar los scripts de la carpeta `pruebas/` y detectar que el dataset de internet repetía fotos de TrashNet, lo que obligó a eliminar duplicados antes de dividir los datos;
  - redactar y ordenar partes del README y de la presentación a partir de las métricas obtenidas.

- **Responsabilidades y decisiones del equipo:** definir el problema y el alcance, elegir el nombre del proyecto, ejecutar los entrenamientos en Google Colab (GPU), probar la app con fotos propias (latas, madera, zapatos), decidir qué cambios mantener y cuáles descartar (por ejemplo, quitar la función de mapa y no usar una regla de confianza que empeoraba la experiencia), revisar las métricas y las limitaciones que se declaran, y publicar el repositorio y la aplicación.

- **Verificación:** las cifras de este README salen de las salidas ejecutadas del notebook (`notebooks/entrenamiento.ipynb`), `modelo/metricas.json` y los scripts de `pruebas/`. Las recomendaciones de Gemini no se evaluaron de forma cuantitativa.

- **Entrenamiento y evaluación:** el entrenamiento de los modelos y el cálculo de las métricas se realizaron en Google Colab.

## Consideraciones éticas
- Prototipo **educativo**: no reemplaza las indicaciones de la municipalidad ni de un punto limpio.
- Con confianza menor a 70 % la app pide revisión humana. En el test, el 94 % de las fotos tiene confianza de 0,7 o más y en ese grupo el accuracy es 95,7 %. Pero el 4,1 % del test (y cerca del 14 % de los residuos en las fotos de Wikimedia Commons) se clasifica mal con confianza alta, y en esos casos la app no avisa.
- No recolectamos fotos de personas ni datos sensibles. Sin embargo, las fotos de internet e ImageNet (ropa, zapatos y objetos cotidianos) pueden incluir personas, por ejemplo con la ropa puesta. Se usan solo con fines educativos y esas fuentes tienen sus propios términos de uso (el dataset de internet no declara licencia y ImageNet tiene términos propios), que conviene revisar antes de cualquier uso distinto.
- Mundo cerrado, mitigado pero no resuelto: antes el modelo no tenía una clase «otro» y clasificó un zapato negro como vidrio con 92,4 % de confianza. Ahora la clase `other` detecta los zapatos reservados (30 de 30) y el 62,5 % de las categorías nunca vistas, pero un objeto muy distinto a los del entrenamiento aún puede clasificarse como un material, y a la inversa, un residuo real puede declararse «no es residuo» (5 % en las fotos de Commons). Por eso se mantiene la revisión humana.
- Etiquetas y datos: el dataset de internet tiene etiquetas ruidosas y fotos casi duplicadas; las métricas sobre esa fuente pueden ser algo optimistas. Por eso se agregaron pruebas con fotos independientes (ver arriba).
- Sesgo: gran parte de las fotos proviene de productos de otros países, y no se midió el rendimiento con envases chilenos ni con fondos muy complejos.
