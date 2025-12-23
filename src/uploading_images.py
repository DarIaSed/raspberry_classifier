import pandas as pd
import os
import requests
import time
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(current_dir)

data_folder = os.path.join(project_dir, "data")
excel_folder = os.path.join(data_folder, "excel_files")
images_folder = os.path.join(data_folder, "images")

os.makedirs(images_folder, exist_ok=True)

species = {
    "Малина_западная": "rubus_occidentalis",
    "Малина_пурпурноплодная": "rubus_phoenicolasius", 
    "Малина_обыкновенная": "rubus_idaeus",
    "Малина_великолепная": "rubus_spectabilis",
    "Малина_мелкоцветковая": "rubus_parviflorus"
}

downloaded_images = []
errors = []


def read_data_file(filepath):
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)


def extract_url_from_row(row):
    if isinstance(row, str) and ',' in row:
        parts = row.split(',')
        if len(parts) > 7:
            return parts[7].strip('"')
    return None


def download_image(url, save_path):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            if 'image' in response.headers.get('Content-Type', ''):
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
    except Exception as e:
        errors.append(f"{url}: {str(e)}")
    
    return False


for file in os.listdir(excel_folder):
    if not (file.endswith('.xlsx') or file.endswith('.csv')):
        continue
    
    print(f"\nФайл: {file}")
    
    name_ru = file.split('.')[0]
    name_en = species.get(name_ru, name_ru)
    
    species_folder = os.path.join(images_folder, name_en)
    os.makedirs(species_folder, exist_ok=True)
    
    df = read_data_file(os.path.join(excel_folder, file))
    print(f"  Найдено {len(df)} записей")
    
    downloaded_count = 0
    
    for i in tqdm(range(len(df)), desc=f"  {name_en}"):
        row = df.iloc[i]
        
        url = None
        
        url = extract_url_from_row(str(row.iloc[0]))
        
        if not url or pd.isna(url):
            continue
        
        url = str(url).strip()
        if not url.startswith('http'):
            url = 'https://' + url
        
        url = url.replace('small', 'medium').replace('large', 'medium')
        
        filename = f"{name_en}_{i+1:04d}.jpg"
        filepath = os.path.join(species_folder, filename)
        
        if os.path.exists(filepath):
            downloaded_count += 1
            downloaded_images.append({
                'вид': name_en,
                'файл': filename,
                'источник': url[:50] + '...' if len(url) > 50 else url
            })
            continue
        
        if download_image(url, filepath):
            downloaded_count += 1
            downloaded_images.append({
                'вид': name_en,
                'файл': filename,
                'источник': url[:50] + '...' if len(url) > 50 else url
            })
        
        time.sleep(0.05)
    
    print(f"  Скачано изображений: {downloaded_count}")

if downloaded_images:    
    print("\nСтатистика по видам:")
    for name in species.values():
        count = len([img for img in downloaded_images if img['вид'] == name])
        if count > 0:
            print(f"  {name}: {count} изображений")