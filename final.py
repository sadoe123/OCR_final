#!/usr/bin/env python
# coding: utf-8

import cv2
import pytesseract
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans
import os
import json
import difflib
import re
import tkinter as tk
from tkinter import filedialog


TITRES_ATTENDUS = [
    "Critères de lancement",
    "Option de consolidation",
    "Critères avancés",
    "Options de génération",
    "Paramètres de lancement","Paramètres avancés"
]
CORRECTIONS_ORTHO = {
    "repports": "rapports",
    "dinstrument": "d'instrument",
    "planification de Lanification": "planification"
}

def nettoyer_texte(text):
    text = text.strip()
    text = re.sub(r"[^\w\sÀ-ÿ'-]", '', text)
    text = re.sub(r"\s{3,}", ' ', text)
    text = text.replace("’", "'").replace("œ", "oe").replace("–", "-")
    text = re.sub(r"\bd(instrument|objet|option|utilisateur|état|élément)\b", lambda m: f"d’{m.group(1)}", text, flags=re.IGNORECASE)
    return text

def corriger_orthographe(text):
    return CORRECTIONS_ORTHO.get(text.strip().lower(), text)

def is_placeholder(text):
    text = text.lower().strip().replace(" ", "")
    placeholders = {'jour/mois/année', 'jour/mois/annee', 'jj/mm/aaaa', 'jjmmaaaa', 'jourmoisannee', 'jourmoisannée'}
    return text in placeholders or not text or len(text) < 2 or 'placeholder' in text

def detecter_placeholder(text):
    placeholders = ["jour/mois/année", "jour/mois/annee", "jj/mm/aaaa", "jjmmaaaa", "Type(s) d'instrument(s)", "Type de cours", "Date", "placeholder"]
    for ph in placeholders:
        if ph.lower() in text.lower():
            return ph.lower()
    return None

def is_title(text):
    text_clean = nettoyer_texte(text).lower()
    for titre in TITRES_ATTENDUS:
        titre_clean = nettoyer_texte(titre).lower()
        if text_clean == titre_clean:
            return True
        if difflib.SequenceMatcher(None, titre_clean, text_clean).ratio() > 0.92:
            return True
    return False

def fusionner_par_placeholder(fused_fields):
    result = []
    i = 0
    while i < len(fused_fields):
        field = fused_fields[i]
        if (i + 1 < len(fused_fields) and
            field['col_num'] == fused_fields[i+1]['col_num'] and
            abs(field['y_top'] - fused_fields[i+1]['y_top']) < 18 and
            field.get('placeholder') and
            field['placeholder'] == fused_fields[i+1].get('placeholder')):
            new_field = field.copy()
            new_field['text'] = field['text'].strip() + " " + fused_fields[i+1]['text'].strip()
            result.append(new_field)
            i += 2
        else:
            result.append(field)
            i += 1
    return result

def fusionner_champs_verticaux(area_fields, seuil_vertical=30):
    fusionnes = []
    area_fields = sorted(area_fields, key=lambda x: (x['col_num'], x['y_top']))
    fragments_suite = ["d'instrument", "d'instruments", "précomptes", "décalés", "rating", 
                      "moyen", "de change", "de cotation", "TCN", "échus"]

    def doit_fusionner(texte1, texte2):
        t1, t2 = texte1.strip().lower(), texte2.strip().lower()
        if ("filtrer sur le type" in t1 and "instrument" in t2) or \
           ("option compta tcn" in t1 and "précomptes" in t2) or \
           ("intérêts échus" in t1 and "décalés" in t2) or \
           ("données" in t1 and "rating" in t2) or \
           ("date taux" in t1 and "change" in t2) or \
           ("place" in t1 and "cotation" in t2):
            return True
        for fragment in fragments_suite:
            if fragment in t2:
                return True
        if any(t1.endswith(word) for word in [' le', ' la', ' les', ' de', ' du', ' sur']):
            return True
        return False

    i = 0
    while i < len(area_fields):
        base = area_fields[i]
        texte_concat = base['text']
        y_top_final = base['y_top']
        j = i + 1
        while j < len(area_fields):
            suivant = area_fields[j]
            if (base['col_num'] == suivant['col_num'] and
                0 < suivant['y_top'] - y_top_final < seuil_vertical and
                doit_fusionner(texte_concat, suivant['text'])):
                texte_concat += " " + suivant['text']
                y_top_final = suivant['y_top']
                j += 1
            else:
                break
        fusionnes.append({
            'text': texte_concat.strip(),
            'col_num': base['col_num'],
            'y_top': base['y_top'],
            'y_center': base['y_center']
        })
        i = j
    return fusionnes


root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Sélectionnez une ou plusieurs images",
    filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")],
    initialdir=os.getcwd()
)

if not file_paths:
    print("❌ Aucune image sélectionnée.")
    exit()

reader = easyocr.Reader(['fr'], gpu=False)
os.makedirs("output", exist_ok=True)

for img_path in file_paths:
    print(f"\n🔍 Traitement : {os.path.basename(img_path)}")
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Erreur : impossible de lire {os.path.basename(img_path)}")
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)

    results = reader.readtext(image, paragraph=False, detail=1)

    x_centers = [[float(np.mean([p[0] for p in bbox]))] for (bbox, _, _) in results]
    if len(x_centers) > 1:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(x_centers)
        column_labels = kmeans.labels_
    else:
        column_labels = [0] * len(x_centers)

    cluster_x_means = {label: np.mean([x_centers[i][0] for i in range(len(x_centers)) if column_labels[i] == label])
                    for label in set(column_labels)}
    sorted_labels = sorted(cluster_x_means, key=cluster_x_means.get)
    label_to_colnum = {label: idx+1 for idx, label in enumerate(sorted_labels)}
    column_numbers = [label_to_colnum[label] for label in column_labels]

    fields_data = []
    for idx, (bbox, text, prob) in enumerate(results):
        if prob < 0.3 or is_placeholder(text):
            continue
        fields_data.append({
            'bbox': bbox,
            'text': corriger_orthographe(nettoyer_texte(text)),
            'col_num': column_numbers[idx],
            'y_center': float(np.mean([p[1] for p in bbox])),
            'y_top': min([p[1] for p in bbox]),
            'idx': idx,
            'placeholder': detecter_placeholder(text)
        })

    fused_fields = []
    used = set()
    for i, f1 in enumerate(fields_data):
        if i in used:
            continue
        group = [f1]
        for j, f2 in enumerate(fields_data):
            if j <= i or j in used:
                continue
            same_col = f1['col_num'] == f2['col_num']
            close_y = abs(f1['y_top'] - f2['y_top']) < 12
            if same_col and close_y:
                group.append(f2)
                used.add(j)
        text_joint = " ".join([f['text'] for f in sorted(group, key=lambda x: x['bbox'][0][0])])
        placeholders = [f.get('placeholder') for f in group if f.get('placeholder')]
        placeholder = placeholders[0] if placeholders and all(ph == placeholders[0] for ph in placeholders) else None
        fused_fields.append({
            'text': text_joint,
            'col_num': f1['col_num'],
            'y_center': np.mean([f['y_center'] for f in group]),
            'y_top': min([f['y_top'] for f in group]),
            'placeholder': placeholder
        })

    fused_fields = fusionner_par_placeholder(fused_fields)

    areas = []
    current_area = None
    for field in fused_fields:
        if is_title(field['text']):
            current_area = {'area': field['text'], 'fields': []}
            areas.append(current_area)
        elif current_area:
            current_area['fields'].append(field)

    output_structure = []
    for idx, area in enumerate(areas):
        champs_fusionnes = fusionner_champs_verticaux(area['fields'])
        fields_sorted = sorted(champs_fusionnes, key=lambda f: (f['y_top'], f['col_num']))
        sort_number = 1
        last_y = None
        tolerance = 10
        for field in fields_sorted:
            if last_y is None or abs(field['y_top'] - last_y) > tolerance:
                if last_y is not None:
                    sort_number += 1
                last_y = field['y_top']
            field['sort_number'] = sort_number

        area_data = {
            "area": area['area'],
            "fields": [
                {
                    'name': f['text'],
                    'sort_number': f['sort_number'],
                    'columnNumber': f['col_num'],
                    'area': f"area{idx + 1}"
                }
                for f in fields_sorted
            ]
        }
        output_structure.append(area_data)


    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    ax.imshow(image_rgb)

    for idx, (bbox, text, prob) in enumerate(results):
        if idx >= len(column_numbers):
            continue
        if is_placeholder(text):
            continue
        col_num = column_numbers[idx]
        color = 'red' if col_num == 1 else 'blue'
        if is_title(text):
            color = 'green'
        poly = patches.Polygon(bbox, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(poly)
        plt.text(bbox[0][0], bbox[0][1]-10, f"{text} (Col{col_num})", color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))

    plt.title(f"Textes détectés et colonnes - {os.path.basename(img_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
json_name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
output_path = os.path.join(OUTPUT_DIR, json_name) 
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_structure, f, indent=2, ensure_ascii=False)
print(f"  JSON exporté dans {output_path}") 
print("\n Traitement terminé pour toutes les images.")
try:
    input("Appuyez sur Entrée pour quitter...")
except EOFError:
    pass
