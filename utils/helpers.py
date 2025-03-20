import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI
import logging
from logging.handlers import RotatingFileHandler

def trace():
    """
    Comprehensive tracing function that writes memory usage statistics
    and DataFrame contents to tmp/output.tmp file for debugging purposes.
    """
    # Get 'objects' from the caller's global namespace
    import inspect
    import os
    frame = inspect.currentframe().f_back
    objects = frame.f_globals
    
    # Create tmp directory if it doesn't exist
    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    # Open the output file in the tmp directory with UTF-8 encoding
    output_path = os.path.join(tmp_dir, "output.tmp")
    with open(output_path, "w", encoding="utf-8") as output_file:
        # The encapsulated inner functions... 
        def _write_memory_usage(objs):
            """Inner function to write memory usage information"""
            output_file.write("\n===== MEMORY USAGE STATISTICS =====\n")
            total_memory = 0
            
            for obj_name, obj in objs.items():
                memory = sys.getsizeof(obj) / (1024 * 1024)
                output_file.write(f"Object '{obj_name}' size: {memory:.2f} MB\n")
                total_memory += memory
            output_file.write(f"Total memory consumption: {total_memory:.2f} MB\n")
        
        def _write_df_head_and_tail(objs):
            """Inner function to write DataFrame head and tail"""
            output_file.write("\n===== DATAFRAME CONTENTS =====\n")
            
            # Save the current Dataframe display settings (Col1, Col2, ... ColN)
            original_max_columns = pd.get_option('display.max_columns')
            original_width = pd.get_option('display.width')
            
            try:
                # Set display options to show all Dataframe's columns
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)  # Adjust width to avoid wrapping
                
                for obj_name, obj in objs.items():
                    # Only process objects that are DataFrames
                    if isinstance(obj, pd.DataFrame):
                        output_file.write(f"\n{'='*50}\n")
                        output_file.write(f"DataFrame: {obj_name}\n")
                        output_file.write(f"Shape: {obj.shape}\n")
                        output_file.write(f"\n--- HEAD (5 first rows) ---\n")
                        output_file.write(f"{obj.head(5)}\n")
                        output_file.write(f"\n--- TAIL (5 last rows) ---\n")
                        output_file.write(f"{obj.tail(5)}\n")
                        output_file.write(f"{'='*50}\n\n")
            finally:
                # Restore Dataframe's original display settings
                pd.set_option('display.max_columns', original_max_columns)
                pd.set_option('display.width', original_width)
        
        # Call the inner functions
        _write_memory_usage(objects)
        _write_df_head_and_tail(objects)


def _setup_logger():
    # Joyuela: Logger configuration with file rotation
    logger = logging.getLogger("ErrorLogger")
    logger.setLevel(logging.ERROR)

    handler = RotatingFileHandler("Misc/Logs/main.py.log", maxBytes=500000, backupCount=3)
    handler.setFormatter(logging.Formatter("="*60 + "\n" + "%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    return logger

# Make the logger available at module level for importing by other modules
logger = _setup_logger()
