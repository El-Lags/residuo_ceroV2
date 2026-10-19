"""Descarga fotos de Wikimedia Commons (licencias abiertas) para probar los modelos con imágenes independientes.

Uso:  python pruebas/descargar_fotos_commons.py <carpeta_de_salida>
Las fotos NO se incluyen en el repositorio; la licencia de cada una queda en manifiesto.json."""
import json, os, random, re, sys, time
import requests

destino = sys.argv[1]
os.makedirs(destino, exist_ok=True)
API = "https://commons.wikimedia.org/w/api.php"
H = {"User-Agent": "ResiduoCero-pruebas/1.0 (proyecto educativo UDD; contacto: vicente.lagunass@gmail.com)"}

# grupo -> (clase esperada del modelo, categorías de Commons, cuántas fotos)
GRUPOS = {
    "latas_metal":  ("metal",     ["Beverage cans", "Soft drink cans", "Beer cans", "Aluminium cans"], 48),
    "botellas_plastico": ("plastic", ["Plastic bottles", "PET bottles", "Soft drink bottles"], 48),
    "vidrio":       ("glass",     ["Glass bottles", "Beer bottles", "Wine bottles", "Glass jars"], 48),
    "cajas_carton": ("cardboard", ["Cardboard boxes", "Corrugated fiberboard"], 48),
    "diarios_papel": ("paper",    ["Newspapers", "Magazines"], 40),
    "basura_calle": ("trash",     ["Litter"], 30),
    "restos_comida": ("trash",    ["Food waste"], 30),
    "madera":       ("other",     ["Tree trunks", "Firewood"], 40),
    "zapatos":      ("other",     ["Shoes"], 20),
    "mochilas":     ("other",     ["Backpacks"], 20),
    "laptops":      ("other",     ["Laptops"], 20),
    "sillas":       ("other",     ["Chairs"], 20),
    "gatos":        ("other",     ["Cats"], 20),
    "bicicletas":   ("other",     ["Bicycles"], 20),
    "flores":       ("other",     ["Flowers"], 20),
}
rng = random.Random(11)

def titulos(cat):
    out, cont = [], {}
    while True:
        r = requests.get(API, params={"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}", "cmtype": "file",
                                      "cmlimit": "500", "format": "json", **cont}, headers=H, timeout=30).json()
        out += [m["title"] for m in r["query"]["categorymembers"]]
        if "continue" in r and len(out) < 1500: cont = r["continue"]
        else: return out

manifiesto = []
for grupo, (clase, cats, n) in GRUPOS.items():
    todos = []
    for c in cats:
        todos += [t for t in titulos(c) if re.search(r"\.(jpe?g|png)$", t, re.I)]
        time.sleep(0.3)
    todos = sorted(set(todos)); rng.shuffle(todos)
    carpeta = os.path.join(destino, grupo); os.makedirs(carpeta, exist_ok=True)
    guardadas = 0
    for i in range(0, len(todos), 40):
        if guardadas >= n: break
        lote = todos[i:i + 40]
        r = requests.get(API, params={"action": "query", "titles": "|".join(lote), "prop": "imageinfo", "iiprop": "url|mime|extmetadata",
                                      "iiurlwidth": "512", "format": "json"}, headers=H, timeout=60).json()
        for p in r.get("query", {}).get("pages", {}).values():
            if guardadas >= n: break
            info = (p.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if not url or info.get("mime") not in ("image/jpeg", "image/png"): continue
            try:
                b = requests.get(url, headers=H, timeout=60)
                if b.status_code != 200 or len(b.content) < 3000: continue
            except Exception:
                continue
            ext = "png" if info.get("mime") == "image/png" else "jpg"
            nombre = f"{grupo}_{guardadas:03d}.{ext}"
            with open(os.path.join(carpeta, nombre), "wb") as f: f.write(b.content)
            lic = (info.get("extmetadata") or {}).get("LicenseShortName", {}).get("value", "")
            manifiesto.append({"archivo": f"{grupo}/{nombre}", "grupo": grupo, "esperada": clase, "titulo": p.get("title"), "licencia": lic})
            guardadas += 1
            time.sleep(0.25)
    print(f"{grupo:18s} {guardadas:3d} fotos (clase esperada: {clase})", flush=True)
json.dump(manifiesto, open(os.path.join(destino, "manifiesto.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("total:", len(manifiesto))
