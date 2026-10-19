"""Residuo Cero · clasificador de residuos con IA.

Flujo: imagen -> EfficientNet-B0 (clase + confianza) -> resultado visual -> LLM (Gemini) -> recomendación.
"""
import json
import os

import requests
import streamlit as st
import timm
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------- configuración
st.set_page_config(page_title="Residuo Cero", page_icon="♻️", layout="wide")

RAIZ = os.path.dirname(os.path.abspath(__file__))
RUTA_PESOS = os.path.join(RAIZ, "modelo", "modelo_efficientnet.pt")
RUTA_CLASES = os.path.join(RAIZ, "modelo", "clases.json")
RUTA_METRICAS = os.path.join(RAIZ, "modelo", "metricas.json")
CARPETA_EJEMPLOS = os.path.join(RAIZ, "ejemplos")

UMBRAL_CONFIANZA = 0.70  # bajo este valor se pide revisión humana (ver sección 6.1 del notebook)

# nombre en español, ícono y contenedor sugerido (Chile) para cada clase del dataset
INFO_CLASES = {
    "cardboard": ("Cartón", "📦", "contenedor azul (papel y cartón)"),
    "glass": ("Vidrio", "🍾", "campana verde de vidrio"),
    "metal": ("Metal", "🥫", "contenedor amarillo (envases y latas)"),
    "paper": ("Papel", "📄", "contenedor azul (papel y cartón)"),
    "plastic": ("Plástico", "🧴", "contenedor amarillo (envases plásticos)"),
    "trash": ("Basura no reciclable", "🗑️", "basura general"),
    "other": ("No parece un residuo", "🚫", "no corresponde a ningún contenedor de reciclaje"),
}
# nombres cortos para los gráficos (los largos se cortan en el eje)
NOMBRE_GRAFICO = {"trash": "Basura", "other": "Otro (no residuo)"}


def etiqueta_grafico(clase):
    return NOMBRE_GRAFICO.get(clase) or INFO_CLASES.get(clase, (clase,))[0]


# un color por material, para la tarjeta de resultado y las barras
COLOR_CLASE = {
    "cardboard": "#FB8C00",
    "glass": "#14B8A6",
    "metal": "#6366F1",
    "paper": "#3B82F6",
    "plastic": "#F59E0B",
    "trash": "#EF4444",
    "other": "#A855F7",
}


def barras_html(valores, formato):
    """Barras horizontales de colores (valores entre 0 y 1), ordenadas de mayor a menor."""
    filas = []
    for clase, v in sorted(valores.items(), key=lambda par: -par[1]):
        color = COLOR_CLASE.get(clase, "#2e7d32")
        icono = INFO_CLASES.get(clase, ("", "❓", ""))[1]
        filas.append(
            f'<div class="fila"><div class="fila-nombre">{icono} {etiqueta_grafico(clase)}</div>'
            f'<div class="pista"><div class="relleno" style="width:{v * 100 if v >= 0.005 else 0:.1f}%;background:{color}"></div></div>'
            f'<div class="fila-valor">{formato(v)}</div></div>'
        )
    return '<div class="barras">' + "".join(filas) + "</div>"


# ---------------------------------------------------------------- estilo
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, button {font-family: 'Poppins', 'Segoe UI', sans-serif;}
    .stApp {background: linear-gradient(180deg, #ecfff3 0%, #e6f7ff 55%, #f6efff 100%);}
    .block-container {padding-top: 4.2rem;}

    /* banner principal */
    .hero {background: radial-gradient(circle at 92% 18%, rgba(255,255,255,.28) 0, rgba(255,255,255,0) 32%),
                       radial-gradient(circle at 6% 110%, rgba(255,255,255,.22) 0, rgba(255,255,255,0) 38%),
                       linear-gradient(120deg, #00c853 0%, #00bfa5 48%, #2979ff 100%);
           border-radius: 28px; padding: 2.2rem 2.6rem; color: #fff; margin-bottom: 1rem;
           box-shadow: 0 14px 34px rgba(0, 150, 110, .30);}
    .hero-titulo {font-size: 3rem; font-weight: 800; line-height: 1.1; letter-spacing: -.5px;}
    .hero-sub {font-size: 1.15rem; margin: .5rem 0 1.1rem; max-width: 46rem; opacity: .96;}
    .pill {display: inline-block; background: rgba(255,255,255,.22); border: 1px solid rgba(255,255,255,.55);
           border-radius: 999px; padding: .32rem .95rem; margin: 0 .5rem .4rem 0; font-size: .92rem; font-weight: 600;}
    .aviso {background: #fff7d6; border: 1px solid #ffe08a; border-radius: 16px; padding: .75rem 1.1rem;
            margin: .4rem 0 1.2rem; color: #6b4e00;}

    /* títulos de sección */
    h2, h3 {color: #0b6e4f !important; font-weight: 700 !important;}

    /* tarjeta de resultado */
    .tarjeta {display: flex; align-items: center; gap: 1.2rem; background: #fff; border-radius: 24px;
              padding: 1.3rem 1.5rem; margin: .3rem 0 1rem; border-left: 10px solid var(--c);
              background: linear-gradient(135deg, color-mix(in srgb, var(--c) 16%, #fff) 0%, #fff 70%);
              box-shadow: 0 10px 26px rgba(0, 0, 0, .10);}
    .tarjeta-icono {font-size: 3.2rem; width: 5.2rem; height: 5.2rem; display: flex; align-items: center;
                    justify-content: center; background: #fff; border-radius: 50%; flex-shrink: 0;
                    box-shadow: 0 4px 14px rgba(0, 0, 0, .12);}
    .clase {font-size: 1.9rem; font-weight: 800; color: #1f2937; line-height: 1.15;}
    .chip {display: inline-block; background: var(--c); color: #fff; border-radius: 999px;
           padding: .22rem .9rem; margin: .45rem 0 .35rem; font-weight: 700; font-size: .95rem;}
    .destino {color: #4b5563; font-size: .98rem;}

    /* barras de colores */
    .barras {margin: .4rem 0 1.3rem;}
    .fila {display: flex; align-items: center; gap: .8rem; margin: .5rem 0;}
    .fila-nombre {width: 10.5rem; font-weight: 600; color: #1f2937; font-size: .95rem;}
    .pista {flex: 1; background: rgba(15, 23, 42, .08); border-radius: 999px; height: .85rem; overflow: hidden;}
    .relleno {height: 100%; border-radius: 999px;}
    .fila-valor {width: 4.2rem; text-align: right; font-weight: 700; color: #1f2937; font-size: .95rem;}

    /* tarjetas de métricas */
    [data-testid="stMetric"] {background: #fff; border-radius: 20px; padding: 1rem 1.3rem;
                              border-top: 6px solid #00c853; box-shadow: 0 8px 22px rgba(0, 0, 0, .08);}
    [data-testid="stMetricValue"] {color: #0b6e4f; font-weight: 800;}

    /* pestañas */
    [role="tab"] p, button[data-baseweb="tab"] p {font-size: 1.15rem; font-weight: 600;}
    [role="tab"][aria-selected="true"] p, button[data-baseweb="tab"][aria-selected="true"] p {color: #0b6e4f;}
    .react-aria-SelectionIndicator {background-color: #00c853 !important; height: 4px !important; border-radius: 4px;}
    div[data-baseweb="tab-highlight"] {background-color: #00c853 !important; height: 4px; border-radius: 4px;}

    /* botón principal */
    .stButton > button[kind="primary"] {background: linear-gradient(120deg, #00c853, #00b0ff); color: #fff;
        border: none; border-radius: 999px; padding: .65rem 1.7rem; font-weight: 700; font-size: 1.02rem;
        box-shadow: 0 8px 20px rgba(0, 176, 255, .35); transition: transform .15s, box-shadow .15s;}
    .stButton > button[kind="primary"]:hover {transform: translateY(-2px); box-shadow: 0 12px 26px rgba(0, 176, 255, .45);}

    /* subida de archivo e imágenes */
    [data-testid="stFileUploader"] section {border: 2px dashed #00c853; border-radius: 20px; background: rgba(255,255,255,.75);}
    [data-testid="stImage"] img {border-radius: 20px; box-shadow: 0 8px 22px rgba(0, 0, 0, .12);}

    /* barra lateral */
    [data-testid="stSidebar"] {background: linear-gradient(190deg, #006e4f 0%, #00897b 55%, #1565c0 100%);}
    [data-testid="stSidebar"] * {color: #fff !important;}
    [data-testid="stSidebar"] hr {border-color: rgba(255,255,255,.35);}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {opacity: 1;}
    .lat-logo {font-size: 3.4rem; line-height: 1; margin-top: .4rem;}
    .lat-titulo {font-size: 1.9rem; font-weight: 800; line-height: 1.1; margin: .2rem 0 .5rem;}
    .lat-frase {font-size: 1.02rem; line-height: 1.45; opacity: .95;}
    .lat-seccion {font-size: .8rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
                  margin: 1.7rem 0 .7rem; opacity: .85;}
    .paso {display: flex; gap: .8rem; align-items: center; background: rgba(255,255,255,.16);
           border: 1px solid rgba(255,255,255,.28); border-radius: 18px; padding: .75rem .9rem; margin-bottom: .6rem;
           font-size: .95rem; line-height: 1.35;}
    .paso-num {flex-shrink: 0; width: 2.1rem; height: 2.1rem; border-radius: 50%; background: #fff;
               color: #0b6e4f !important; font-weight: 800; display: flex; align-items: center; justify-content: center;}
    .tip {background: rgba(255,255,255,.12); border-radius: 999px; padding: .4rem .9rem; margin-bottom: .45rem;
          font-size: .93rem;}
    .bin {display: flex; align-items: center; gap: .65rem; margin-bottom: .5rem; font-size: .95rem;}
    .punto {width: 1rem; height: 1rem; border-radius: 50%; flex-shrink: 0; border: 2px solid rgba(255,255,255,.85);}
    .lat-tec {font-size: .82rem; line-height: 1.5; opacity: .85; border-top: 1px solid rgba(255,255,255,.3); padding-top: .7rem;}
    .lat-pie {margin-top: 1.2rem; font-size: .78rem; line-height: 1.4; opacity: .8;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- modelo
@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo():
    """Carga EfficientNet-B0 con los pesos entrenados en el notebook."""
    with open(RUTA_CLASES, encoding="utf-8") as f:
        clases = json.load(f)
    modelo = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(clases))
    modelo.load_state_dict(torch.load(RUTA_PESOS, map_location="cpu"))
    modelo.eval()
    return modelo, clases


PREPROCESO = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # estadísticas de ImageNet
])


def clasificar(imagen, modelo, clases):
    """Devuelve un diccionario {clase: probabilidad} ordenado de mayor a menor."""
    x = PREPROCESO(imagen.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(modelo(x), dim=1)[0].tolist()
    return dict(sorted(zip(clases, probs), key=lambda par: -par[1]))


def cargar_metricas():
    """Métricas medidas en el conjunto de prueba (las guarda el notebook)."""
    try:
        with open(RUTA_METRICAS, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------- LLM (Gemini)
def pedir_recomendacion(clase, confianza):
    """Envía solo texto (clase y confianza) a Gemini y devuelve la recomendación."""
    try:                                   # si no existe secrets.toml, Streamlit lanza un error
        secretos = dict(st.secrets)
    except Exception:
        secretos = {}
    clave = secretos.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not clave:
        return "⚠️ Falta configurar GEMINI_API_KEY como secreto de la aplicación."
    modelo_llm = secretos.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    nombre, _, contenedor = INFO_CLASES.get(clase, (clase, "", "según normativa local"))
    destino = contenedor.split(" (")[0]    # sin el paréntesis, que nombra otros materiales (ej. «papel y cartón»)
    prompt = f"""Eres un asistente educativo sobre reciclaje domiciliario en Chile.
Un modelo de visión clasificó la foto de un residuo así:
- Clase detectada: {nombre}
- Confianza del modelo: {confianza:.0%}
- Destino habitual: {destino}

Responde en español, con un máximo de 120 palabras, usando estas secciones:
1. Cómo preparar el residuo antes de reciclarlo.
2. Dónde depositarlo.
3. Un dato breve sobre su impacto ambiental.

Reglas:
- No saludes ni te presentes: empieza directamente con la sección 1.
- Háblale solo de reciclaje y manejo de residuos; si algo no corresponde, no lo inventes.
- Habla únicamente de {nombre}: no menciones otros materiales ni otros tipos de residuo, ni siquiera como comparación o ejemplo.
- {"La confianza es BAJA: empieza diciendo que la clasificación es incierta y pide al usuario verificar el material a simple vista o consultar la norma local antes de decidir, sin nombrar otros materiales." if confianza < UMBRAL_CONFIANZA else "No cuestiones la clasificación: responde sobre este material."}
- Es información educativa: no reemplaza las indicaciones de tu municipalidad ni de un punto limpio."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo_llm}:generateContent"
    try:
        for intento in range(2):  # un reintento si Google está lento o saturado
            try:
                r = requests.post(
                    url,
                    headers={"x-goog-api-key": clave, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=45,
                )
                if r.status_code in (429, 500, 503) and intento == 0:
                    continue
                break
            except requests.Timeout:
                if intento == 1:
                    raise
        r.raise_for_status()
        partes = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p["text"] for p in partes if "text" in p and not p.get("thought"))
    except requests.HTTPError as e:  # Google respondió con un error: mostramos su código y mensaje
        try:
            detalle = e.response.json()["error"]["message"][:200]
        except Exception:
            detalle = e.response.text[:200]
        return f"⚠️ No se pudo obtener la recomendación (error {e.response.status_code}): {detalle}"
    except Exception as e:  # la app no debe caerse si el LLM falla
        return f"⚠️ No se pudo obtener la recomendación ({type(e).__name__}). Intenta de nuevo."


# ---------------------------------------------------------------- barra lateral
metricas_lateral = cargar_metricas()
exactitud = f"<br>Accuracy en test: {metricas_lateral['accuracy']:.1%}" if metricas_lateral else ""

with st.sidebar:
    st.markdown(
        '<div class="lat-logo">♻️</div><div class="lat-titulo">Residuo Cero</div>'
        '<div class="lat-frase">¿No sabes dónde botar algo? Sácale una foto y te lo decimos.</div>'
        '<div class="lat-seccion">¿Cómo funciona?</div>'
        '<div class="paso"><span class="paso-num">1</span><div><b>Sube una foto</b><br>de lo que quieres botar.</div></div>'
        '<div class="paso"><span class="paso-num">2</span><div><b>La IA lo mira</b><br>y te dice de qué material es.</div></div>'
        '<div class="paso"><span class="paso-num">3</span><div><b>Te decimos dónde va</b><br>y, si quieres, cómo dejarlo listo.</div></div>'
        '<div class="lat-seccion">Para una buena foto</div>'
        '<div class="tip">☝️ Un solo objeto a la vez</div>'
        '<div class="tip">⬜ Con fondo liso, si puedes</div>'
        '<div class="tip">💡 Con buena luz y de cerca</div>'
        '<div class="lat-seccion">¿A qué contenedor va?</div>'
        '<div class="bin"><span class="punto" style="background:#3B82F6"></span>Azul: papel y cartón</div>'
        '<div class="bin"><span class="punto" style="background:#FACC15"></span>Amarillo: latas y plásticos</div>'
        '<div class="bin"><span class="punto" style="background:#22C55E"></span>Verde: vidrio</div>'
        '<div class="bin"><span class="punto" style="background:#9CA3AF"></span>Gris: basura común</div>'
        '<div class="lat-seccion">Detalles técnicos</div>'
        '<div class="lat-tec">Modelo: EfficientNet-B0 (timm) ajustado con TrashNet, fotos de internet e ImageNet '
        f'(clase «otro»).{exactitud}<br>Recomendaciones: Google Gemini.</div>'
        '<div class="lat-pie">Prototipo educativo · Proyecto del ramo IA Generativa para la Gestión Industrial, UDD.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- encabezado
st.markdown(
    '<div class="hero"><div class="hero-titulo">♻️ Residuo Cero</div>'
    '<div class="hero-sub">Sube la foto de un residuo y descubre de qué material es y en qué contenedor va.</div>'
    '<span class="pill">📷 Sube una foto</span><span class="pill">🧠 La IA identifica el material</span>'
    '<span class="pill">💬 Consejos para reciclar</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="aviso">🎓 <b>Prototipo educativo.</b> Los resultados pueden equivocarse; '
    'ante la duda, consulta las indicaciones de tu municipalidad o punto limpio.</div>',
    unsafe_allow_html=True,
)

try:
    modelo, clases = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar el modelo ({type(e).__name__}). Revisa que la carpeta `modelo/` tenga los pesos y las clases.")
    st.stop()

pestana_app, pestana_modelo = st.tabs(["📷 Clasificar", "📊 Sobre el modelo"])

# ---------------------------------------------------------------- pestaña 1: clasificar
with pestana_app:
    col_entrada, col_resultado = st.columns(2, gap="large")

    with col_entrada:
        st.subheader("1. Elige una imagen")
        archivo = st.file_uploader("Sube una foto (JPG o PNG)", type=["jpg", "jpeg", "png", "webp"])

        ejemplos = sorted(
            f for f in (os.listdir(CARPETA_EJEMPLOS) if os.path.isdir(CARPETA_EJEMPLOS) else [])
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        ejemplo = st.selectbox("…o prueba con un ejemplo", ["—"] + ejemplos)

        imagen = None
        id_imagen = None                      # identifica la imagen para no mezclar recomendaciones entre fotos
        if archivo is not None:
            imagen = Image.open(archivo)
            id_imagen = f"subida:{getattr(archivo, 'file_id', archivo.name)}"
        elif ejemplo != "—":
            imagen = Image.open(os.path.join(CARPETA_EJEMPLOS, ejemplo))
            id_imagen = f"ejemplo:{ejemplo}"
        if imagen is not None:
            st.image(imagen, caption="Imagen a clasificar", width="stretch")

    with col_resultado:
        st.subheader("2. Resultado")
        if imagen is None:
            st.info("Sube una imagen o elige un ejemplo para empezar.")
        else:
            probs = clasificar(imagen, modelo, clases)
            clase, confianza = next(iter(probs.items()))
            nombre, icono, contenedor = INFO_CLASES.get(clase, (clase, "❓", "—"))

            st.markdown(
                f'<div class="tarjeta" style="--c:{COLOR_CLASE.get(clase, "#2e7d32")}">'
                f'<div class="tarjeta-icono">{icono}</div><div>'
                f'<div class="clase">{nombre}</div>'
                f'<span class="chip">Confianza de esta imagen: {confianza:.1%}</span>'
                f'<div class="destino">Destino habitual: {contenedor}</div></div></div>',
                unsafe_allow_html=True,
            )
            if confianza < UMBRAL_CONFIANZA:
                st.warning(
                    "⚠️ **Confianza baja.** El modelo no está seguro: revisa el material a simple vista "
                    "(revisión humana) antes de decidir dónde depositarlo."
                )

            if clase == "other":
                st.info(
                    "🚫 **Esto no parece un residuo reciclable** (por ejemplo, calzado, ropa, muebles o electrónicos). "
                    "Si de verdad quieres desecharlo, consulta en tu municipalidad o punto limpio cómo hacerlo."
                )
            else:
                st.caption(
                    "ℹ️ El modelo distingue 6 materiales y la clase «no parece un residuo». Con objetos muy distintos "
                    "a los que vio al entrenar aún puede equivocarse, incluso con confianza alta."
                )

            st.markdown("**Probabilidad de cada clase**")
            st.markdown(barras_html(probs, lambda v: f"{v:.1%}"), unsafe_allow_html=True)

            if clase != "other":                       # si no es un residuo, no hay nada que recomendar
                st.markdown("---")
                st.subheader("3. Recomendación con IA")
                recomendaciones = st.session_state.setdefault("recomendaciones", {})   # {id de imagen: texto}
                if st.button("💬 Generar recomendación", type="primary"):
                    with st.spinner("Consultando a Gemini…"):
                        recomendaciones[id_imagen] = pedir_recomendacion(clase, confianza)
                if id_imagen in recomendaciones:       # solo se muestra la de la imagen que está en pantalla
                    st.markdown(recomendaciones[id_imagen])
                    st.caption("Generado por IA. Es orientación educativa y puede contener errores.")

# ---------------------------------------------------------------- pestaña 2: sobre el modelo
with pestana_modelo:
    st.subheader("Desempeño del modelo en el conjunto de prueba")
    st.caption(
        "Estas cifras se midieron **una sola vez** con imágenes que el modelo nunca vio al entrenar, pero de las mismas fuentes "
        "que el entrenamiento (TrashNet, internet e ImageNet): **con fotos de otras fuentes el acierto baja** (ver las limitaciones abajo). "
        "No son la confianza de la foto que subiste: esa es distinta para cada imagen."
    )
    metricas = cargar_metricas()
    if metricas is None:
        st.info("Aún no hay métricas: ejecuta el notebook y copia `modelo/metricas.json` al repositorio.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metricas['accuracy']:.1%}")
        c2.metric("Accuracy balanceada", f"{metricas['accuracy_balanceada']:.1%}")
        c3.metric("Imágenes de prueba", metricas["n_test"])
        st.markdown("**Recall por clase** (qué porcentaje de cada material detecta el modelo)")
        st.markdown(barras_html(metricas["recall_por_clase"], lambda v: f"{v:.0%}"), unsafe_allow_html=True)
        st.markdown(
            f"**Línea base:** CLIP sin entrenar obtiene {metricas['clip_accuracy']:.1%} de accuracy "
            "con las mismas imágenes de prueba."
        )
        if "accuracy_por_fuente" in metricas:
            nombres_fuente = {"trashnet": "TrashNet", "web": "internet", "imagenet": "ImageNet"}
            partes = " · ".join(f"{nombres_fuente.get(f, f)}: {a:.0%}" for f, a in metricas["accuracy_por_fuente"].items())
            st.markdown(f"**Accuracy por fuente de las fotos de prueba:** {partes}.")
        if "otro_zapatos_nunca_vistos" in metricas:
            st.markdown(
                "**Objetos que no son residuos** (fotos que el modelo nunca vio al entrenar): "
                f"zapatos detectados como «No parece un residuo»: {metricas['otro_zapatos_nunca_vistos']:.0%} · "
                f"otras categorías de objetos: {metricas['otro_categorias_nunca_vistas']:.0%}."
            )

    st.markdown("### Limitaciones y ética")
    st.markdown(
        "- El modelo se entrenó con unas 7.900 fotos (TrashNet, internet e ImageNet). Con fotos muy distintas a esas, por ejemplo con fondos complejos o mala luz, puede fallar.\n"
        "- **Con fotos nuevas de otras fuentes, el acierto en residuos ronda el 80 %,** bastante menos que las cifras de arriba. Las latas enteras se confunden a veces con vidrio o plástico, y las flores o plantas pueden salir como basura.\n"
        "- **Objetos que no son residuos:** antes el modelo no tenía una clase «otro» y clasificó un zapato negro como vidrio con 92 % de confianza. Ahora la clase «No parece un residuo» detecta los zapatos reservados y cerca del 60 % de las categorías de objetos que nunca vio, pero un objeto muy distinto a los del entrenamiento aún puede clasificarse como un material, incluso con confianza alta, y a la inversa, un residuo real puede declararse «no es residuo».\n"
        "- Las fotos de entrenamiento de internet e ImageNet pueden incluir personas con la ropa puesta; se usan solo con fines educativos.\n"
        "- Equivocarse en un residuo contaminado (p. ej. mandar basura al contenedor de reciclaje) **contamina** todo el lote: por eso se pide revisión humana cuando la confianza es baja.\n"
        "- La recomendación la escribe un modelo de lenguaje a partir de la clase y la confianza; no ve la imagen."
    )
