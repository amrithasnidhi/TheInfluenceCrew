import json

with open('influencer_analysis.ipynb', 'r', encoding='utf-8') as f:
    n = json.load(f)

for cell in n['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'pca_data = df[pca_cols].dropna()' in line:
                source[i] = line.replace('df[pca_cols]', "df[pca_cols + ['purchase_ord']]")
                print(f"Modified line in cell index {n['cells'].index(cell)}: {source[i].strip()}")

with open('influencer_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(n, f, indent=1)
