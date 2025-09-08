#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set

def analyze_line_types(data_root: str) -> Dict[str, int]:
    """
    Analyze all annotation files to discover unique line types.
    
    Args:
        data_root: Root directory containing SoccerNet GSR data
        
    Returns:
        Dictionary mapping line types to their occurrence count
    """
    data_root = Path(data_root)
    line_types = defaultdict(int)
    
    # Find all JSON annotation files
    json_files = list(data_root.glob("*/Labels-GameState.json"))
    print(f"Found {len(json_files)} annotation files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                annotation = json.load(f)
            
            # Process all annotations
            for ann in annotation.get("annotations", []):
                if ann.get("supercategory") == "pitch":
                    lines = ann.get("lines", {})
                    for line_name in lines.keys():
                        line_types[line_name] += 1
                        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return dict(line_types)

def main():
    # Analyze both train and valid directories
    train_root = "/Users/boyan531/Documents/football/SoccerNet/SN-GSR-2025/train"
    
    print("=== ANALYZING PITCH LINE TYPES ===")
    print(f"Scanning: {train_root}")
    
    train_line_types = analyze_line_types(train_root)
    
    print(f"\n=== DISCOVERED LINE TYPES ===")
    print(f"Total unique line types: {len(train_line_types)}")
    
    # Sort by frequency
    sorted_types = sorted(train_line_types.items(), key=lambda x: x[1], reverse=True)
    
    for i, (line_type, count) in enumerate(sorted_types):
        print(f"{i+1:2d}. '{line_type}': {count} occurrences")
    
    # Generate class mapping
    print(f"\n=== SUGGESTED CLASS MAPPING ===")
    print("# Background class")
    print("LINE_CLASS_MAPPING = {")
    print("    'background': 0,  # No line/background")
    
    for i, (line_type, count) in enumerate(sorted_types):
        clean_name = line_type.lower().replace(" ", "_")
        print(f"    '{clean_name}': {i+1},  # {line_type}")
    
    print("}")
    print(f"\nTOTAL CLASSES: {len(train_line_types) + 1} (including background)")

if __name__ == "__main__":
    main()