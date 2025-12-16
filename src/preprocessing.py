import pandas as pd
from bs4 import BeautifulSoup
import os

def load_and_parse_data(filepath):
    """
    Membaca file raw HTML dan mengubahnya menjadi DataFrame Pandas.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan di: {filepath}")

    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')
    
    # Mencari tabel data (berdasarkan struktur HTML actors.html)
    # File tersebut menggunakan format tabel <tr><td>...
    rows = []
    tables = soup.find_all('table')
    
    for table in tables:
        for tr in table.find_all('tr'):
            cells = tr.find_all('td')
            if len(cells) > 0:
                # Mengambil text dari setiap sel dan membersihkannya
                row = [cell.get_text().strip() for cell in cells]
                # Pastikan row memiliki cukup kolom (minimal sampai Role Type)
                if len(row) >= 8: 
                    rows.append(row)

    # Membuat DataFrame
    # Kolom berdasarkan header di file HTML: stage, dow, birth, giv, gen, dob, dod, type, orig, pict, notes
    columns = ['stage_name', 'dow', 'family_name', 'given_name', 'gender', 'dob', 'dod', 'role_type', 'origin', 'pict', 'notes']
    
    # Karena data di HTML kadang tidak konsisten jumlah kolomnya, kita ambil 8 kolom utama saja yang penting
    df = pd.DataFrame(rows)
    df = df.iloc[:, :9] # Ambil sampai Origin
    df.columns = columns[:9]
    
    return df

def clean_data(df):
    """
    Membersihkan data: Handle missing values, encoding gender, menyederhanakan role.
    Target: Memprediksi apakah aktor tersebut 'Lead' (Pemeran Utama) atau 'Support' berdasarkan fitur lain.
    """
    # 1. Bersihkan Role Type
    # Kita buat binary classification: 1 jika 'lead' atau 'hero', 0 jika 'support', 'character', dll.
    def categorize_role(role):
        role = str(role).lower()
        if 'lead' in role or 'hero' in role or 'star' in role:
            return 1
        else:
            return 0

    df['is_lead'] = df['role_type'].apply(categorize_role)

    # 2. Bersihkan Gender (M/F)
    df = df[df['gender'].isin(['M', 'F'])] # Hapus yang aneh-aneh
    
    # 3. Bersihkan Origin (Ambil kode negara utama, misal \Am -> Am)
    df['origin'] = df['origin'].astype(str).str.replace(r'[^a-zA-Z]', '', regex=True)
    df = df[df['origin'] != ''] # Hapus yang kosong

    # 4. Pilih fitur yang akan dipakai
    # Fitur: Gender, Origin
    # Target: is_lead
    final_df = df[['gender', 'origin', 'is_lead']].copy()
    
    print(f"Data cleaning selesai. Jumlah data bersih: {len(final_df)} baris.")
    return final_df

if __name__ == "__main__":
    # Path relative
    input_path = os.path.join('data', 'raw', 'actors.html')
    output_path = os.path.join('data', 'processed', 'actors_clean.csv')
    
    # Pastikan folder output ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Memulai proses ETL...")
    try:
        df_raw = load_and_parse_data(input_path)
        df_clean = clean_data(df_raw)
        df_clean.to_csv(output_path, index=False)
        print(f"Data berhasil disimpan di {output_path}")
    except Exception as e:
        print(f"Terjadi error: {e}")