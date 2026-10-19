"""Evalúa un modelo (y otro de comparación) con las fotos de Commons elegidas a ojo.

Uso:  python pruebas/evaluar_fotos_commons.py <carpeta_fotos> <carpeta_modelo_anterior> <carpeta_modelo_nuevo> <hoja_de_fallas.png>
Las fotos 'curadas' (objeto claro como protagonista) se eligieron antes de ver ninguna predicción."""
import collections, json, os, sys
import numpy as np, timm, torch
from PIL import Image, ImageDraw
from torchvision import transforms

carpeta, dir_v1, dir_v2, salida = sys.argv[1:5]
NOMB = ["cardboard", "glass", "metal", "other", "paper", "plastic", "trash"]
NOM = {"cardboard": "cartón", "glass": "vidrio", "metal": "metal", "other": "otro", "paper": "papel", "plastic": "plástico", "trash": "basura"}
PRE = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def cargar(c):
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7)
    m.load_state_dict(torch.load(os.path.join(c, "modelo_efficientnet.pt"), map_location="cpu")); m.eval(); return m
M1, M2 = cargar(dir_v1), cargar(dir_v2)

# fotos elegidas a ojo (objeto claro como protagonista), ANTES de ver ninguna predicción; None = todas
CURADAS = {
    "latas_metal": [1, 5, 7, 12, 15, 16, 19, 20, 24, 25, 26, 30, 35, 36, 39, 43, 44],
    "botellas_plastico": [0, 1, 2, 4, 5, 6, 8, 19, 21, 23, 33, 36, 40, 46],
    "vidrio": [1, 3, 4, 5, 7, 8, 10, 13, 16, 18, 19, 20, 21, 30, 31, 32, 35, 36, 44, 46],
    "cajas_carton": [1, 3, 5, 7, 8, 9, 14, 15, 19, 31, 32, 34, 35, 36, 43, 46],
    "diarios_papel": [1, 8, 10, 16, 19, 26, 28, 29, 31, 32],
    "restos_comida": [3, 4, 5, 22, 25],
    "madera": [2, 5, 7, 12, 14, 21, 22, 26, 27, 29, 31, 33, 38, 39],
    "zapatos": None, "mochilas": None, "laptops": None, "sillas": None, "gatos": None, "bicicletas": None, "flores": None,
}
man = json.load(open(os.path.join(carpeta, "manifiesto.json"), encoding="utf-8"))
sel = []
for m in man:
    g = m["grupo"]
    if g not in CURADAS: continue
    idx = int(m["archivo"].split("_")[-1].split(".")[0])
    if CURADAS[g] is None or idx in CURADAS[g]: sel.append(m)
print(f"fotos evaluadas: {len(sel)} (de {len(man)} descargadas; las demás eran escenas, escaneos o paisajes)\n")

ims = [Image.open(os.path.join(carpeta, m["archivo"])).convert("RGB") for m in sel]
def probs(M):
    out = []
    for i in range(0, len(ims), 32):
        x = torch.stack([PRE(im) for im in ims[i:i + 32]])
        with torch.no_grad(): out.append(torch.softmax(M(x), 1).numpy())
    return np.concatenate(out)
P1, P2 = probs(M1), probs(M2)
esp = np.array([NOMB.index(m["esperada"]) for m in sel]); grupos = np.array([m["grupo"] for m in sel])

print(f"{'grupo':18s} {'n':>3s} | {'ACTUAL acierta':>14s} | {'NUEVO v2 acierta':>16s} | el nuevo responde (top 3)")
tot = {"waste": [[], []], "otro": [[], []]}
for g in CURADAS:
    i = np.where(grupos == g)[0]
    if not len(i): continue
    a1, a2 = np.mean(P1[i].argmax(1) == esp[i]), np.mean(P2[i].argmax(1) == esp[i])
    dist = collections.Counter(NOM[NOMB[k]] for k in P2[i].argmax(1)).most_common(3)
    print(f"{g:18s} {len(i):3d} | {a1:14.0%} | {a2:16.0%} | " + ", ".join(f"{a} {b / len(i):.0%}" for a, b in dist))
    if g == "restos_comida": continue
    k = "otro" if NOMB[esp[i[0]]] == "other" else "waste"
    tot[k][0] += list(P1[i].argmax(1) == esp[i]); tot[k][1] += list(P2[i].argmax(1) == esp[i])
for k, nombre in (("waste", "RESIDUOS (latas, plástico, vidrio, cartón, papel)"), ("otro", "NO RESIDUOS (madera, zapatos, mochilas, laptops, sillas, gatos, bicicletas, flores)")):
    print(f"\n{nombre}: {len(tot[k][0])} fotos · ACTUAL acierta {np.mean(tot[k][0]):.1%} · NUEVO v2 acierta {np.mean(tot[k][1]):.1%}")

w = np.array([g not in ("restos_comida",) and NOMB[e] != "other" for g, e in zip(grupos, esp)])
for nombre, P in (("ACTUAL", P1), ("NUEVO v2", P2)):
    mal = (P.argmax(1) != esp) & w; conf = P.max(1)
    print(f"{nombre:9s}: en residuos, se equivoca con confianza >= 70 % (la app no avisa) en {np.mean((mal & (conf >= 0.7))[w]):.1%} de las fotos; "
          f"dice «no es residuo» en {np.mean((P.argmax(1) == NOMB.index('other'))[w]):.1%}")

# lista de fallas del v2 y hoja de contacto
fallas = [(j, sel[j]["grupo"], NOMB[P2[j].argmax()], P2[j].max(), NOMB[esp[j]]) for j in range(len(sel)) if P2[j].argmax() != esp[j] and sel[j]["grupo"] != "restos_comida"]
print(f"\nFALLAS DEL NUEVO v2 ({len(fallas)} fotos):")
for j, g, p, c, e in fallas[:60]:
    print(f"   {sel[j]['archivo']:34s} era {NOM[e]:8s} -> dijo {NOM[p]:8s} {c:.0%}")
L, COLS = 190, 6
filas = (len(fallas) + COLS - 1) // COLS
hoja = Image.new("RGB", (COLS * L, max(filas, 1) * (L + 30)), "white"); d = ImageDraw.Draw(hoja)
for k, (j, g, p, c, e) in enumerate(fallas[:60]):
    im = ims[j].copy(); im.thumbnail((L - 4, L - 4)); x, y = (k % COLS) * L, (k // COLS) * (L + 30)
    hoja.paste(im, (x + 2, y + 30)); d.text((x + 4, y + 2), f"era {NOM[e]}", fill="black"); d.text((x + 4, y + 14), f"dijo {NOM[p]} {c:.0%}", fill="red")
hoja.save(salida)
