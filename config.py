import os
from pathlib import Path

# Obtener la ruta base del proyecto (ruta relativa)
BASE_DIR = Path(__file__).parent

# Usar variables de entorno con valores predeterminados
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(BASE_DIR, 'datasets'))
LOG_DIR = os.environ.get('LOG_DIR', os.path.join(BASE_DIR, 'misc', 'logs'))
TEMP_DIR = os.environ.get('TEMP_DIR', os.path.join(BASE_DIR, 'tmp'))

# Definir rutas específicas usando pathlib
PARQUET_FILES = {
    'user_reviews': Path(DATA_DIR) / 'Clean_Parquet_Data_Steam' / 'Clean_australian_user_reviews_FE.parquet',
    'steam_games': Path(DATA_DIR) / 'Clean_Parquet_Data_Steam' / 'Clean_output_steam_games.parquet',
    'user_items': Path(DATA_DIR) / 'Clean_Parquet_Data_Steam' / 'Clean_australian_users_items.parquet'
}

LOG_FILE = Path(LOG_DIR) / 'main.py.log'
TEMP_OUTPUT = Path(TEMP_DIR) / 'output.tmp'
