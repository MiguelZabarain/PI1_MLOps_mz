import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI
import logging
from logging.handlers import RotatingFileHandler

def trace():
    """
    Comprehensive tracing function that prints memory usage statistics
    and displays DataFrame contents for debugging purposes.
    """
    # Get 'objects' from the caller's global namespace
    import inspect
    frame = inspect.currentframe().f_back
    objects = frame.f_globals
    
    # The encapsulated inner functions... 
    def _print_memory_usage(objs):
        """Inner function to print memory usage information"""
        print("\n===== MEMORY USAGE STATISTICS =====")
        total_memory = 0
        
        for obj_name, obj in objs.items():
            memory = sys.getsizeof(obj) / (1024 * 1024)
            print(f"Object '{obj_name}' size: {memory:.2f} MB")
            total_memory += memory
        print(f"Total memory consumption: {total_memory:.2f} MB")
    
    def _print_df_head_and_tail(objs):
        """Inner function to print DataFrame head and tail"""
        print("\n===== DATAFRAME CONTENTS =====")
        
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
                    print(f"\n{'='*50}")
                    print(f"DataFrame: {obj_name}")
                    print(f"Shape: {obj.shape}")
                    print(f"\n--- HEAD (5 first rows) ---")
                    print(obj.head(5))
                    print(f"\n--- TAIL (5 last rows) ---")
                    print(obj.tail(5))
                    print(f"{'='*50}\n")
        finally:
            # Restore Dataframe's original display settings
            pd.set_option('display.max_columns', original_max_columns)
            pd.set_option('display.width', original_width)
    
    # Call the inner functions
    _print_memory_usage(objects)
    _print_df_head_and_tail(objects)


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
