import os
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import io
import base64
import html # For escaping HTML characters in data attributes
import re

# --- Matplotlib Setup (for backend graph generation) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- CWIValidatorTester Class ---
class CWIValidatorTester:
    def __init__(self):
        self.sqfeet_ranges = {}
        self.vectorizers = {}
        self.particular_vectors = {}
        self.particular_names = {}
        self.training_stats = {}

    def extract_sqfeet_from_filename(self, filename, mode='quantity'):
        """Extract square feet range from filename based on mode"""
        filename_lower = filename.lower()

        # Check for explicit range patterns first (work for both modes)
        # 250 sqft ranges (amount mode)
        amount_patterns = [
            (r'500\s*-?\s*1000', '500-1000'),
            (r'1001\s*-?\s*1500', '1001-1500'),
            (r'1501\s*-?\s*2000', '1501-2000'),
            (r'2001\s*-?\s*2500', '2001-2500'),
            (r'2501\s*-?\s*3000', '2501-3000'),
            (r'3001\s*-?\s*3500', '3001-3500'),
            (r'3501\s*-?\s*4000', '3501-4000'),
            (r'4001\s*-?\s*4500', '4001-4500'),
            (r'4501\s*-?\s*5000', '4501-5000')
        ]
        # 500 sqft ranges (quantity mode)
        quantity_patterns = [
            (r'500\s*-?\s*1000', '500-1000'),
            (r'1001\s*-?\s*1500', '1001-1500'),
            (r'1501\s*-?\s*2000', '1501-2000'),
            (r'2001\s*-?\s*2500', '2001-2500'),
            (r'2501\s*-?\s*3000', '2501-3000'),
            (r'3001\s*-?\s*3500', '3001-3500'),
            (r'3501\s*-?\s*4000', '3501-4000'),
            (r'4001\s*-?\s*4500', '4001-4500'),
            (r'4501\s*-?\s*5000', '4501-5000')
        ]

        # Try explicit patterns first
        for pattern, range_key in amount_patterns:
            if re.search(pattern, filename_lower):
                return range_key

        for pattern, range_key in quantity_patterns:
            if re.search(pattern, filename_lower):
                return range_key

        # Check for 5000+ pattern
        if re.search(r'5000\+|above\s*5000|more\s*than\s*5000', filename_lower):
            return '5000+'

        # Fallback: Extract number and determine range based on mode
        numbers = re.findall(r'\d+', filename_lower)
        if numbers:
            for num_str in numbers:
                try:
                    num = int(num_str)

                    if mode == 'amount':
                        # Use 500 sqft ranges for AMOUNT mode
                        if 500 <= num <= 1000:
                            return '500-1000'
                        elif 1001 <= num <= 1500:
                            return '1001-1500'
                        elif 1501 <= num <= 2000:
                            return '1501-2000'
                        elif 2001 <= num <= 2500:
                            return '2001-2500'
                        elif 2501 <= num <= 3000:
                            return '2501-3000'
                        elif 3001 <= num <= 3500:
                            return '3001-3500'
                        elif 3501 <= num <= 4000:
                            return '3501-4000'
                        elif 4001 <= num <= 4500:
                            return '4001-4500'
                        elif 4501 <= num <= 5000:
                            return '4501-5000'
                        elif num > 5000:
                            return '5000+'
                    elif mode == 'rate':
                        # Use 250 sqft ranges for RATE mode
                        if 500 <= num <= 750:
                            return '500-750'
                        elif 751 <= num <= 1000:
                            return '751-1000'
                        elif 1001 <= num <= 1250:
                            return '1001-1250'
                        elif 1251 <= num <= 1500:
                            return '1251-1500'
                        elif 1501 <= num <= 1750:
                            return '1501-1750'
                        elif 1751 <= num <= 2000:
                            return '1751-2000'
                        elif 2001 <= num <= 2250:
                            return '2001-2250'
                        elif 2251 <= num <= 2500:
                            return '2251-2500'
                        elif 2501 <= num <= 2750:
                            return '2501-2750'
                        elif 2751 <= num <= 3000:
                            return '2751-3000'
                        elif 3001 <= num <= 3250:
                            return '3001-3250'
                        elif 3251 <= num <= 3500:
                            return '3251-3500'
                        elif 3501 <= num <= 3750:
                            return '3501-3750'
                        elif 3751 <= num <= 4000:
                            return '3751-4000'
                        elif 4001 <= num <= 4250:
                            return '4001-4250'
                        elif 4251 <= num <= 4500:
                            return '4251-4500'
                        elif 4501 <= num <= 4750:
                            return '4501-4750'
                        elif 4751 <= num <= 5000:
                            return '4751-5000'
                        elif num > 5000:
                            return '5000+'
                    else:
                        # Use 500 sqft ranges (default for quantity mode)
                        if 500 <= num <= 1000:
                            return '500-1000'
                        elif 1001 <= num <= 1500:
                            return '1001-1500'
                        elif 1501 <= num <= 2000:
                            return '1501-2000'
                        elif 2001 <= num <= 2500:
                            return '2001-2500'
                        elif 2501 <= num <= 3000:
                            return '2501-3000'
                        elif 3001 <= num <= 3500:
                            return '3001-3500'
                        elif 3501 <= num <= 4000:
                            return '3501-4000'
                        elif 4001 <= num <= 4500:
                            return '4001-4500'
                        elif 4501 <= num <= 5000:
                            return '4501-5000'
                        elif num > 5000:
                            return '5000+'
                except ValueError:
                    continue

        return None
    def extract_sqfeet_from_filename_regional(self, filename):
        """Extract square feet range for REGIONAL models (500-2500, 2500-5000, 5000+)"""
        filename_lower = filename.lower()

        # Regional patterns - LARGER intervals
        patterns = [
            (r'500\s*-?\s*2500', '500-2500'),
            (r'2500\s*-?\s*5000', '2500-5000'),
            (r'5000\+|above\s*5000|more\s*than\s*5000', '5000+')
        ]

        for pattern, range_key in patterns:
            if re.search(pattern, filename_lower):
                return range_key

        # Fallback: Extract number and determine regional range
        numbers = re.findall(r'\d+', filename_lower)
        if numbers:
            for num_str in numbers:
                try:
                    num = int(num_str)
                    if 500 <= num <= 2500:
                        return '500-2500'
                    elif 2501 <= num <= 5000:
                        return '2500-5000'
                    elif num > 5000:
                        return '5000+'
                except ValueError:
                    continue

        return None

    def clean_text(self, text):
        """Clean and normalize text for better matching"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def extract_quantity(self, qty_value):
        """Extract numeric quantity from various formats"""
        if pd.isna(qty_value):
            return 0
        qty_str = str(qty_value).strip()
        numbers = re.findall(r'\d+\.?\d*', qty_str)
        if numbers:
            return float(numbers[0])
        return 0

    def extract_amount(self, amount_value):
        """Extract numeric amount from various formats"""
        if pd.isna(amount_value):
            return 0
        amount_str = str(amount_value).strip()
        # Remove currency symbols and commas
        amount_str = amount_str.replace('₹', '').replace('Rs', '').replace(',', '').replace('INR', '').strip()
        # Handle negative values
        amount_str = amount_str.replace('(', '-').replace(')', '')
        numbers = re.findall(r'-?\d+\.?\d*', amount_str)
        if numbers:
            return float(numbers[0])
        return 0

    def normalize_unit(self, unit_value):
        """Normalize unit values to check if it's number-based"""
        if pd.isna(unit_value):
            return None
        unit_str = str(unit_value).strip().upper()
        # Number-based units
        number_units = ['NOS', 'NO', 'NOS.', 'NO.', 'NUMBER', 'NUMBERS', 'QTY', 'QUANTITY']
        for nu in number_units:
            if nu in unit_str:
                return 'NUMBER_BASED'
        return unit_str

    def is_number_based_unit(self, unit_value):
        """Check if unit is number-based"""
        normalized = self.normalize_unit(unit_value)
        return normalized == 'NUMBER_BASED'

    def round_quantity_if_number_based(self, quantity, unit_value):
        """Round quantity if unit is number-based"""
        if self.is_number_based_unit(unit_value):
            return round(quantity)
        return quantity

    def infer_unit_from_particular(self, particular_text):
        """Use AI logic to infer unit from particular description"""
        particular_lower = str(particular_text).lower()

        # Keywords indicating number-based quantities
        number_keywords = ['panel', 'door', 'window', 'unit', 'piece', 'item', 'set', 
                           'fixture', 'board', 'sheet', 'nos', 'number', 'quantity']

        # Keywords indicating area-based quantities
        area_keywords = ['sqft', 'sq ft', 'square feet', 'sft', 'area', 'flooring', 
                         'painting', 'plastering', 'tiling']

        # Keywords indicating length-based quantities
        length_keywords = ['rft', 'running feet', 'linear', 'length', 'pipe', 'wire', 
                           'cable', 'railing']

        # Check for number-based
        for keyword in number_keywords:
            if keyword in particular_lower:
                return 'NOS'

        # Check for area-based
        for keyword in area_keywords:
            if keyword in particular_lower:
                return 'SFT'

        # Check for length-based
        for keyword in length_keywords:
            if keyword in particular_lower:
                return 'RFT'

        # Default to NOS if uncertain
        return 'NOS'

    def load_model(self, model_path):
        """Load trained model from file"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.sqfeet_ranges = model_data['sqfeet_ranges']
            self.vectorizers = model_data['vectorizers']
            self.particular_vectors = model_data['particular_vectors']
            self.particular_names = model_data['particular_names']
            self.training_stats = model_data.get('training_stats', {})
            
            print(f"✓ Model loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def find_similar_particular(self, query_particular, sqfeet_range, threshold=0.5):
        """Find most similar particular from training data"""
        if sqfeet_range not in self.vectorizers:
            return None, 0
        
        vectorizer = self.vectorizers[sqfeet_range]
        particular_vectors = self.particular_vectors[sqfeet_range]
        particular_names = self.particular_names[sqfeet_range]
        
        query_vector = vectorizer.transform([self.clean_text(query_particular)])
        similarities = cosine_similarity(query_vector, particular_vectors)[0]
        best_match_idx = np.argmax(similarities)
        
        if similarities[best_match_idx] >= threshold:
            return particular_names[best_match_idx], similarities[best_match_idx]
        
        return None, similarities[best_match_idx]

    def validate_quantity(self, particular, quantity, sqfeet_range, is_regional=False, selected_region=None):
        """Validate quantity for a particular item within specific sq feet range"""
        # Determine which key to use based on what's available in the model
        if 'rate_ranges' in self.sqfeet_ranges.get(sqfeet_range, {}):
            range_key = 'rate_ranges'
        elif 'amount_ranges' in self.sqfeet_ranges.get(sqfeet_range, {}):
            range_key = 'amount_ranges'
        else:
            range_key = 'quantity_ranges'
    
        if sqfeet_range not in self.sqfeet_ranges or not self.sqfeet_ranges[sqfeet_range].get(range_key, {}):
            return True, f"No validation rules for sq feet range: {sqfeet_range}", None, None, None, None, None, None, [], []
    
        range_data = self.sqfeet_ranges[sqfeet_range]
        cleaned_particular = self.clean_text(particular)
    
        # ✅ NEW: If regional mode, fetch stats from all 4 regions
        regional_stats = {}
        if is_regional and selected_region:
            regions = ['east', 'west', 'north', 'south']
            for region in regions:
                model_key = f'icici_rate_{region}'
                if model_key in VALIDATORS:
                    regional_validator = VALIDATORS[model_key]
                    if sqfeet_range in regional_validator.sqfeet_ranges:
                        regional_range_data = regional_validator.sqfeet_ranges[sqfeet_range]
                        if range_key in regional_range_data and cleaned_particular in regional_range_data[range_key]:
                            regional_stats[region] = regional_range_data[range_key][cleaned_particular]
    
        # For exact match:
        if cleaned_particular in range_data[range_key]:
            stats = range_data[range_key][cleaned_particular]
    
            threshold_min = stats['overall_min']
            threshold_max = stats['overall_max']
            weighted_avg_min = stats.get('weighted_avg_min', stats['overall_mean'])
            weighted_avg_max = stats.get('weighted_avg_max', stats['overall_mean'])
            min_threshold_files = stats.get('min_threshold_files', [])
            max_threshold_files = stats.get('max_threshold_files', [])
    
            if quantity == 0 and stats.get('zero_allowed', False):
                return True, "Zero quantity allowed", cleaned_particular, 1.0, threshold_min, threshold_max, weighted_avg_min, weighted_avg_max, min_threshold_files, max_threshold_files
    
            unit = stats.get('unit', None)
    
            # Round if unit is number-based
            if unit and self.is_number_based_unit(unit):
                threshold_min = round(threshold_min)
                threshold_max = round(threshold_max)
                weighted_avg_min = round(weighted_avg_min)
                weighted_avg_max = round(weighted_avg_max)
    
            # ✅ NEW: Build validation message with all regional ranges
            # ✅ NEW: Build validation message with all regional ranges
            if is_regional and regional_stats:
                # Create separate messages for each region
                regional_messages = {}
                for region in ['east', 'west', 'north', 'south']:
                    if region in regional_stats:
                        r_stats = regional_stats[region]
                        r_min = r_stats.get('weighted_avg_min', r_stats.get('overall_mean', 0))
                        r_max = r_stats.get('weighted_avg_max', r_stats.get('overall_mean', 0))

                        if unit and self.is_number_based_unit(unit):
                            r_min = round(r_min)
                            r_max = round(r_max)
                            regional_messages[region] = f"{int(r_min)} - {int(r_max)}"
                        else:
                            regional_messages[region] = f"{self.format_indian_currency(r_min)} - {self.format_indian_currency(r_max)}"
                    else:
                        regional_messages[region] = "N/A"

                # ✅ FIX: Validate against SELECTED REGION for is_valid
                if selected_region and selected_region in regional_stats:
                    validation_stats = regional_stats[selected_region]
                    val_min = validation_stats.get('weighted_avg_min', validation_stats['overall_mean'])
                    val_max = validation_stats.get('weighted_avg_max', validation_stats['overall_mean'])

                    if unit and self.is_number_based_unit(unit):
                        is_valid = val_min <= round(quantity) <= val_max
                    else:
                        is_valid = val_min <= quantity <= val_max
                else:
                    is_valid = True  # Default to valid if no region selected

                # Return individual messages instead of combined
                return is_valid, regional_messages, cleaned_particular, 1.0, threshold_min, threshold_max, weighted_avg_min, weighted_avg_max, min_threshold_files, max_threshold_files
    
        # Similar match logic (for when exact match not found)
        similar_particular, similarity = self.find_similar_particular(particular, sqfeet_range)
        if similar_particular:
            stats = range_data[range_key][similar_particular]
    
            threshold_min = stats['overall_min']
            threshold_max = stats['overall_max']
            weighted_avg_min = stats.get('weighted_avg_min', stats['overall_mean'])
            weighted_avg_max = stats.get('weighted_avg_max', stats['overall_mean'])
            min_threshold_files = stats.get('min_threshold_files', [])
            max_threshold_files = stats.get('max_threshold_files', [])
    
            if quantity == 0 and stats.get('zero_allowed', False):
                return True, "Zero quantity allowed (Similar Match)", similar_particular, similarity, threshold_min, threshold_max, weighted_avg_min, weighted_avg_max, min_threshold_files, max_threshold_files
    
            unit = stats.get('unit', None)
    
            if unit and self.is_number_based_unit(unit):
                threshold_min = round(threshold_min)
                threshold_max = round(threshold_max)
                weighted_avg_min = round(weighted_avg_min)
                weighted_avg_max = round(weighted_avg_max)
    
            # ✅ NEW: Build validation message with all regional ranges for similar match
            if is_regional and regional_stats:
                msg_parts = []
                for region in ['east', 'west', 'north', 'south']:
                    if region in regional_stats:
                        r_stats = regional_stats[region]
                        r_min = r_stats.get('weighted_avg_min', r_stats.get('overall_mean', 0))
                        r_max = r_stats.get('weighted_avg_max', r_stats.get('overall_mean', 0))
    
                        if unit and self.is_number_based_unit(unit):
                            r_min = round(r_min)
                            r_max = round(r_max)
                            msg_parts.append(f"{region.capitalize()}: {int(r_min)} - {int(r_max)}")
                        else:
                            msg_parts.append(f"{region.capitalize()}: {self.format_indian_currency(r_min)} - {self.format_indian_currency(r_max)}")
    
                msg = "Regional Ranges (Similar Match):\n" + "\n".join(msg_parts)
            else:
                if unit and self.is_number_based_unit(unit):
                    msg = f"Range (from similar match): {int(weighted_avg_min)} - {int(weighted_avg_max)}"
                else:
                    msg = f"Range (from similar match): {weighted_avg_min:.2f} - {weighted_avg_max:.2f}"
    
            # Validate against selected region for similar match
            if is_regional and selected_region and selected_region in regional_stats:
                validation_stats = regional_stats[selected_region]
                val_min = validation_stats.get('weighted_avg_min', validation_stats['overall_mean'])
                val_max = validation_stats.get('weighted_avg_max', validation_stats['overall_mean'])
    
                if unit and self.is_number_based_unit(unit):
                    is_valid = val_min <= round(quantity) <= val_max
                else:
                    is_valid = val_min <= quantity <= val_max
            else:
                if unit and self.is_number_based_unit(unit):
                    is_valid = weighted_avg_min <= round(quantity) <= weighted_avg_max
                else:
                    is_valid = weighted_avg_min <= quantity <= weighted_avg_max
    
            return is_valid, msg, similar_particular, similarity, threshold_min, threshold_max, weighted_avg_min, weighted_avg_max, min_threshold_files, max_threshold_files
    
        # No match found
        return True, "No matching particular found - automatically approved", None, 0, None, None, None, None, [], []
    def validate_test_file(self, test_file_path, mode='quantity', is_regional=False):
        """Validate test file based on mode (quantity, amount, or rate)"""
        # ✅ FIX: Use regional extraction if it's a regional model
        if is_regional:
            sqfeet_range = self.extract_sqfeet_from_filename_regional(os.path.basename(test_file_path))
            print(f"🗺️  Using REGIONAL range: {sqfeet_range}")
        else:
            sqfeet_range = self.extract_sqfeet_from_filename(os.path.basename(test_file_path), mode=mode)
            print(f"📏 Using STANDARD range: {sqfeet_range}")
    
        if not sqfeet_range or sqfeet_range not in self.sqfeet_ranges:
            return None, f"Could not determine Sq. Feet range from filename or no rules for this range. Available ranges: {list(self.sqfeet_ranges.keys())}"
    
        try:
            df = pd.read_excel(test_file_path, sheet_name='Extracted Data')
    
            # Define expected columns based on mode
            if mode == 'amount':
                expected_cols = ['Sr.No', 'Particulars', 'As Per CWI (Amount)', 'Unit']
                qty_col_name = 'As Per CWI (Amount)'
            elif mode == 'rate':
                expected_cols = ['Sr.No', 'Particulars', 'Rate', 'Unit', 'As Per CWI (Qty)', 'As Per CWI (Amount)']
                qty_col_name = 'Rate'
            else:
                expected_cols = ['Sr.No', 'Particulars', 'As Per CWI (Qty)', 'Unit']
                qty_col_name = 'As Per CWI (Qty)'
    
            col_mapping = {}
    
            # Print available columns for debugging
            print(f"Available columns in Excel: {df.columns.tolist()}")
            print(f"Looking for columns: {expected_cols}")
    
            for expected_col in expected_cols:
                found = False
                for df_col in df.columns:
                    # More aggressive cleaning for matching
                    clean_expected = expected_col.lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
                    clean_df_col = str(df_col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
    
                    # Check if they match
                    if clean_expected == clean_df_col or clean_expected in clean_df_col or clean_df_col in clean_expected:
                        col_mapping[expected_col] = df_col
                        found = True
                        print(f"✓ Matched '{expected_col}' to '{df_col}'")
                        break
    
                if not found:
                    print(f"✗ Could not find column: '{expected_col}'")
                    return None, f"Required column '{expected_col}' could not be found. Available columns: {df.columns.tolist()}"
    
            df_renamed = df.rename(columns={v: k for k, v in col_mapping.items()})
            results = []
    
            # ✅ NEW: Get selected region from Flask session (if available)
            selected_region = None
            try:
                from flask import session
                selected_region = session.get('last_region', '')
                if selected_region:
                    print(f"📍 Selected region from session: {selected_region}")
            except:
                print("⚠️  Could not access Flask session")
    
            for _, row in df_renamed.iterrows():
                if pd.isna(row.get('Particulars')) or str(row.get('Particulars')).strip() == '': 
                    continue
    
                particular_val = row.get('Particulars')
                qty_val = row.get(qty_col_name, 0)
                unit_val = row.get('Unit', '')
    
                # Infer unit if missing
                if pd.isna(unit_val) or str(unit_val).strip() == '':
                    unit_val = self.infer_unit_from_particular(particular_val)
    
                # Extract and round quantity if number-based
                qty = self.extract_quantity(qty_val)
                qty = self.round_quantity_if_number_based(qty, unit_val)
    
                # ✅ NEW: Skip rows with zero amounts in AMOUNT mode only
                if mode == 'amount' and qty == 0:
                    print(f"  ⚠️  Skipping particular with zero amount: {particular_val}")
                    continue
    
                # ✅ FIX: Pass is_regional AND selected_region to validate_quantity
                validation_result = self.validate_quantity(
                    particular_val, qty, sqfeet_range, is_regional=is_regional, selected_region=selected_region
                )
                
                # ✅ NEW: Handle regional validation results differently
                if is_regional and isinstance(validation_result[1], dict):
                    # Regional mode returns dictionary of messages
                    is_valid = validation_result[0]
                    regional_msgs = validation_result[1]
                    matched = validation_result[2]
                    sim = validation_result[3]
                    threshold_min = validation_result[4]
                    threshold_max = validation_result[5]
                    weighted_avg_min = validation_result[6]
                    weighted_avg_max = validation_result[7]
                    min_threshold_files = validation_result[8]
                    max_threshold_files = validation_result[9]
                else:
                    # Standard mode returns single message string
                    is_valid, msg, matched, sim, threshold_min, threshold_max, weighted_avg_min, weighted_avg_max, min_threshold_files, max_threshold_files = validation_result
                    regional_msgs = None
    
                # Build result dict based on mode
                if mode == 'rate':
                    rate_value = qty
                    qty_value = self.extract_quantity(row.get('As Per CWI (Qty)', 0))
                    amount_value = self.extract_quantity(row.get('As Per CWI (Amount)', 0))
                    qty_value = self.round_quantity_if_number_based(qty_value, unit_val)
    
                    # ✅ NEW: Build result dict with regional columns if applicable
                    if is_regional and regional_msgs:
                        result_dict = {
                            'Sr.No': row.get('Sr.No', ''), 
                            'Original_Particular': str(particular_val),
                            'Rate': rate_value,
                            'Unit': unit_val,
                            'Quantity': qty_value,
                            'Amount': amount_value,
                            'Sq_Feet_Range': sqfeet_range, 
                            'Is_Valid': is_valid,
                            'Validation_Message_East': regional_msgs.get('east', 'N/A'),
                            'Validation_Message_West': regional_msgs.get('west', 'N/A'),
                            'Validation_Message_North': regional_msgs.get('north', 'N/A'),
                            'Validation_Message_South': regional_msgs.get('south', 'N/A'),
                            'Selected_Region': selected_region,
                            'Matched_Particular': matched, 
                            'Similarity_Score': sim,
                            'Threshold_Min': threshold_min,
                            'Threshold_Max': threshold_max,
                            'Weighted_Avg_Min': weighted_avg_min,
                            'Weighted_Avg_Max': weighted_avg_max,
                            'Min_Threshold_Files': min_threshold_files,
                            'Max_Threshold_Files': max_threshold_files
                        }
                    else:
                        result_dict = {
                            'Sr.No': row.get('Sr.No', ''), 
                            'Original_Particular': str(particular_val),
                            'Rate': rate_value,
                            'Unit': unit_val,
                            'Quantity': qty_value,
                            'Amount': amount_value,
                            'Sq_Feet_Range': sqfeet_range, 
                            'Is_Valid': is_valid,
                            'Validation_Message': msg, 
                            'Matched_Particular': matched, 
                            'Similarity_Score': sim,
                            'Threshold_Min': threshold_min,
                            'Threshold_Max': threshold_max,
                            'Weighted_Avg_Min': weighted_avg_min,
                            'Weighted_Avg_Max': weighted_avg_max,
                            'Min_Threshold_Files': min_threshold_files,
                            'Max_Threshold_Files': max_threshold_files
                        }
                        
                elif mode == 'amount':
                    # ✅ NEW: Build result dict with regional columns if applicable
                    if is_regional and regional_msgs:
                        result_dict = {
                            'Sr.No': row.get('Sr.No', ''), 
                            'Original_Particular': str(particular_val),
                            'Amount': qty,
                            'Unit': unit_val,
                            'Sq_Feet_Range': sqfeet_range, 
                            'Is_Valid': is_valid,
                            'Validation_Message_East': regional_msgs.get('east', 'N/A'),
                            'Validation_Message_West': regional_msgs.get('west', 'N/A'),
                            'Validation_Message_North': regional_msgs.get('north', 'N/A'),
                            'Validation_Message_South': regional_msgs.get('south', 'N/A'),
                            'Selected_Region': selected_region,
                            'Matched_Particular': matched, 
                            'Similarity_Score': sim,
                            'Threshold_Min': threshold_min,
                            'Threshold_Max': threshold_max,
                            'Weighted_Avg_Min': weighted_avg_min,
                            'Weighted_Avg_Max': weighted_avg_max,
                            'Min_Threshold_Files': min_threshold_files,
                            'Max_Threshold_Files': max_threshold_files
                        }
                    else:
                        result_dict = {
                            'Sr.No': row.get('Sr.No', ''), 
                            'Original_Particular': str(particular_val),
                            'Amount': qty,
                            'Unit': unit_val,
                            'Sq_Feet_Range': sqfeet_range, 
                            'Is_Valid': is_valid,
                            'Validation_Message': msg, 
                            'Matched_Particular': matched, 
                            'Similarity_Score': sim,
                            'Threshold_Min': threshold_min,
                            'Threshold_Max': threshold_max,
                            'Weighted_Avg_Min': weighted_avg_min,
                            'Weighted_Avg_Max': weighted_avg_max,
                            'Min_Threshold_Files': min_threshold_files,
                            'Max_Threshold_Files': max_threshold_files
                        }
                        
                else:  # quantity mode
                    # ✅ NEW: Build result dict with regional columns if applicable
                    if is_regional and regional_msgs:
                        result_dict = {
                            'Sr.No': row.get('Sr.No', ''), 
                            'Original_Particular': str(particular_val),
                            'Quantity': qty,
                            'Unit': unit_val,
                            'Sq_Feet_Range': sqfeet_range, 
                            'Is_Valid': is_valid,
                            'Validation_Message_East': regional_msgs.get('east', 'N/A'),
                            'Validation_Message_West': regional_msgs.get('west', 'N/A'),
                            'Validation_Message_North': regional_msgs.get('north', 'N/A'),
                            'Validation_Message_South': regional_msgs.get('south', 'N/A'),
                            'Selected_Region': selected_region,
                            'Matched_Particular': matched, 
                            'Similarity_Score': sim,
                            'Threshold_Min': threshold_min,
                            'Threshold_Max': threshold_max,
                            'Weighted_Avg_Min': weighted_avg_min,
                            'Weighted_Avg_Max': weighted_avg_max,
                            'Min_Threshold_Files': min_threshold_files,
                            'Max_Threshold_Files': max_threshold_files
                        }
                    else:
                        result_dict = {
                            'Sr.No': row.get('Sr.No', ''), 
                            'Original_Particular': str(particular_val),
                            'Quantity': qty,
                            'Unit': unit_val,
                            'Sq_Feet_Range': sqfeet_range, 
                            'Is_Valid': is_valid,
                            'Validation_Message': msg, 
                            'Matched_Particular': matched, 
                            'Similarity_Score': sim,
                            'Threshold_Min': threshold_min,
                            'Threshold_Max': threshold_max,
                            'Weighted_Avg_Min': weighted_avg_min,
                            'Weighted_Avg_Max': weighted_avg_max,
                            'Min_Threshold_Files': min_threshold_files,
                            'Max_Threshold_Files': max_threshold_files
                        }
    
                results.append(result_dict)
    
            return results, None
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"An unexpected error occurred while processing the Excel file: {e}"


    def format_indian_currency(self, value):
        """Format number to Indian currency style"""
        try:
            if pd.isna(value) or value == 'NA': return str(value)
            val_float = float(value)
            s = "{:.2f}".format(val_float)
            parts = s.split('.')
            integer_part = parts[0]
            decimal_part = parts[1]
            
            if len(integer_part) > 3:
                last_three = integer_part[-3:]
                rest = integer_part[:-3]
                rest = re.sub(r"\B(?=(\d{2})+(?!\d))", ",", rest)
                formatted_int = rest + "," + last_three
            else:
                formatted_int = integer_part
                
            return f"{formatted_int}.{decimal_part}"
        except:
            return str(value)

# --- CWIGroupedValidator Class for Grouped Per Sq.Feet Validation ---
class CWIGroupedValidator:
    def __init__(self):
        # Store data segregated by sq feet ranges (250 sqft intervals)
        self.sqfeet_ranges = {
            '500-750': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '751-1000': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '1001-1250': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '1251-1500': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '1501-1750': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '1751-2000': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '2001-2250': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '2251-2500': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '2501-2750': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '2751-3000': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '3001-3250': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '3251-3500': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '3501-3750': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '3751-4000': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '4001-4250': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '4251-4500': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '4501-4750': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '4751-5000': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)},
            '5000+': {'grouped_ranges': {}, 'particulars_data': defaultdict(list)}
        }
        self.vectorizers = {}
        self.particular_vectors = {}
        self.particular_names = {}
        self.training_stats = {}
        self.unknown_particulars = set()
        self.all_groups = set()
        self.new_groups_found = set()
        
        # Standard grouped categories with their keywords
        self.known_groups = {
            'Civil Works': ['civil', 'civl'],
            'Carpentry Works': ['carpentry', 'carpentary', 'carpenter', 'carpentory'],
            'Non-Interior Works': ['non-int', 'non int', 'nonint', 'non interior', 'noninterior'],
            'Electrical Works': ['electrical', 'electric', 'electricl'],
            'Additional Works': ['additional', 'addi', 'additonal', 'aditional'],
            'Reimbursement of Vendor Elements': ['reimbursement', 'reimbrsement', 'reimburse', 'vendor', 'reimbursment'],
            'Electricity of Diesel Expense': ['electricity of diesel', 'diesel expense', 'electricity diesel', 'diesel']
        }
   
    def extract_sqfeet_from_filename(self, filename):
        """Extract square feet range from filename (250 sqft intervals)"""
        filename_lower = filename.lower()

        # Patterns for 250 sqft intervals
        patterns = [
            (r'500\s*-?\s*750', '500-750'),
            (r'751\s*-?\s*1000', '751-1000'),
            (r'1001\s*-?\s*1250', '1001-1250'),
            (r'1251\s*-?\s*1500', '1251-1500'),
            (r'1501\s*-?\s*1750', '1501-1750'),
            (r'1751\s*-?\s*2000', '1751-2000'),
            (r'2001\s*-?\s*2250', '2001-2250'),
            (r'2251\s*-?\s*2500', '2251-2500'),
            (r'2501\s*-?\s*2750', '2501-2750'),
            (r'2751\s*-?\s*3000', '2751-3000'),
            (r'3001\s*-?\s*3250', '3001-3250'),
            (r'3251\s*-?\s*3500', '3251-3500'),
            (r'3501\s*-?\s*3750', '3501-3750'),
            (r'3751\s*-?\s*4000', '3751-4000'),
            (r'4001\s*-?\s*4250', '4001-4250'),
            (r'4251\s*-?\s*4500', '4251-4500'),
            (r'4501\s*-?\s*4750', '4501-4750'),
            (r'4751\s*-?\s*5000', '4751-5000')
        ]

        for pattern, range_key in patterns:
            if re.search(pattern, filename_lower):
                return range_key

        # Check for 5000+ pattern
        if re.search(r'5000\+|above\s*5000|more\s*than\s*5000', filename_lower):
            return '5000+'

        # Numeric logic for 250 sqft intervals
        numbers = re.findall(r'\d+', filename_lower)
        if numbers:
            for num_str in numbers:
                num = int(num_str)
                
                if 500 <= num <= 750:
                    return '500-750'
                elif 751 <= num <= 1000:
                    return '751-1000'
                elif 1001 <= num <= 1250:
                    return '1001-1250'
                elif 1251 <= num <= 1500:
                    return '1251-1500'
                elif 1501 <= num <= 1750:
                    return '1501-1750'
                elif 1751 <= num <= 2000:
                    return '1751-2000'
                elif 2001 <= num <= 2250:
                    return '2001-2250'
                elif 2251 <= num <= 2500:
                    return '2251-2500'
                elif 2501 <= num <= 2750:
                    return '2501-2750'
                elif 2751 <= num <= 3000:
                    return '2751-3000'
                elif 3001 <= num <= 3250:
                    return '3001-3250'
                elif 3251 <= num <= 3500:
                    return '3251-3500'
                elif 3501 <= num <= 3750:
                    return '3501-3750'
                elif 3751 <= num <= 4000:
                    return '3751-4000'
                elif 4001 <= num <= 4250:
                    return '4001-4250'
                elif 4251 <= num <= 4500:
                    return '4251-4500'
                elif 4501 <= num <= 4750:
                    return '4501-4750'
                elif 4751 <= num <= 5000:
                    return '4751-5000'
                elif num > 5000:
                    return '5000+'

        return None
   
    def clean_text(self, text):
        """Clean and normalize text for better matching"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def standardize_group_name(self, particular_text):
        """Standardize particular name - match to known groups or create new group"""
        if pd.isna(particular_text) or str(particular_text).strip() == '':
            return None
    
        cleaned_text = self.clean_text(particular_text)
        original_text = str(particular_text).strip()
    
        # First, check against known groups
        for standard_name, keywords in self.known_groups.items():
            for keyword in keywords:
                if keyword in cleaned_text:
                    self.all_groups.add(standard_name)
                    return standard_name
    
        # If no match found, create a new group name from the original text
        new_group_name = self._create_group_name(original_text)
        self.all_groups.add(new_group_name)
        self.new_groups_found.add(new_group_name)
    
        return new_group_name

    def _create_group_name(self, text):
        """Create a clean group name from text"""
        # Remove 'total of', 'total', prefixes
        text = re.sub(r'^total\s+of\s+', '', text.lower())
        text = re.sub(r'^total\s+', '', text.lower())

        # Capitalize each word
        words = text.split()
        capitalized = ' '.join(word.capitalize() for word in words)

        # Add 'Works' suffix if not present
        if 'work' not in capitalized.lower():
            capitalized += ' Works'
        elif 'works' not in capitalized.lower():
            capitalized = capitalized.replace('Work', 'Works')

        return capitalized
    def format_indian_currency(self, value):
        """Format number to Indian currency style (e.g., 7,00,909.11)"""
        try:
            if pd.isna(value): return "0.00"
            val_float = float(value)
            s = "{:.2f}".format(val_float)
            parts = s.split('.')
            integer_part = parts[0]
            decimal_part = parts[1]
            
            if len(integer_part) > 3:
                last_three = integer_part[-3:]
                rest = integer_part[:-3]
                # Regex to add commas every 2 digits for the Indian style
                rest = re.sub(r"\B(?=(\d{2})+(?!\d))", ",", rest)
                formatted_int = rest + "," + last_three
            else:
                formatted_int = integer_part
                
            return f"{formatted_int}.{decimal_part}"
        except:
            return str(value)
    def extract_amount(self, amount_value):
        """Extract numeric amount from various formats"""
        if pd.isna(amount_value):
            return 0
       
        amount_str = str(amount_value).strip()
        # Remove currency symbols and commas
        amount_str = amount_str.replace('₹', '').replace('Rs', '').replace(',', '').replace('INR', '').strip()
        # Handle negative values
        amount_str = amount_str.replace('(', '-').replace(')', '')
        numbers = re.findall(r'-?\d+\.?\d*', amount_str)
        if numbers:
            return float(numbers[0])
        return 0
    
    def find_similar_group(self, query_particular, sqfeet_range, threshold=0.5):
        """Find most similar group from standard groups within specific sq feet range"""
        if sqfeet_range not in self.vectorizers or sqfeet_range not in self.particular_vectors:
            return None, 0
        
        vectorizer = self.vectorizers[sqfeet_range]
        particular_vectors = self.particular_vectors[sqfeet_range]
        particular_names = self.particular_names[sqfeet_range]
        
        query_cleaned = self.clean_text(query_particular)
        query_vector = vectorizer.transform([query_cleaned])
        
        similarities = cosine_similarity(query_vector, particular_vectors)[0]
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity >= threshold:
            return particular_names[best_match_idx], best_similarity
        
        return None, best_similarity
   
    def validate_grouped_data(self, group_name, cwi_amount, rate, sqfeet_range):
        """Validate CWI amount and rate for a grouped category within specific sq feet range"""
        if sqfeet_range not in self.sqfeet_ranges:
            return {
                'cwi_valid': False,
                'rate_valid': False,
                'message': f"Invalid sq feet range: {sqfeet_range}"
            }
        
        range_data = self.sqfeet_ranges[sqfeet_range]
        
        # Standardize the group name
        standard_group = self.standardize_group_name(group_name)
        
        if standard_group is None:
            # Try similarity matching
            similar_group, similarity = self.find_similar_group(group_name, sqfeet_range)
            if similar_group:
                standard_group = similar_group
            else:
                return {
                    'cwi_valid': False,
                    'rate_valid': False,
                    'message': f"Unknown group: '{group_name}' - not found in standard groups"
                }
        
        # Check if this group exists in training data for this sq feet range
        if standard_group not in range_data['grouped_ranges']:
            return {
                'cwi_valid': False,
                'rate_valid': False,
                'message': f"Group '{standard_group}' not found in training data for {sqfeet_range} range"
            }
        
        group_stats = range_data['grouped_ranges'][standard_group]
        cwi_stats = group_stats['cwi_amount_stats']
        rate_stats = group_stats['rate_stats']
        
        # Validate CWI amount
        cwi_valid = False
        cwi_message = ""
        # --- UPDATED VALIDATION LOGIC WITH FORMATTING ---
        if cwi_amount == 0 and cwi_stats['zero_allowed']:
            cwi_valid = True
            cwi_message = f"Zero CWI amount allowed ({cwi_stats['zero_count']}/{cwi_stats['total_count']} cases)"
        elif cwi_stats['validation_range_lower'] <= cwi_amount <= cwi_stats['validation_range_upper']:
            cwi_valid = True
            cwi_message = f"Within range: ₹{self.format_indian_currency(cwi_stats['validation_range_lower'])} - ₹{self.format_indian_currency(cwi_stats['validation_range_upper'])}"
        else:
            cwi_message = f"Outside range: ₹{self.format_indian_currency(cwi_stats['validation_range_lower'])} - ₹{self.format_indian_currency(cwi_stats['validation_range_upper'])}"
        
        # Validate Rate
        rate_valid = False
        rate_message = ""
        
        if rate == 0 and rate_stats['zero_allowed']:
            rate_valid = True
            rate_message = f"Zero rate allowed ({rate_stats['zero_count']}/{rate_stats['total_count']} cases)"
        elif rate_stats['validation_range_lower'] <= rate <= rate_stats['validation_range_upper']:
            rate_valid = True
            rate_message = f"Within range: ₹{self.format_indian_currency(rate_stats['validation_range_lower'])} - ₹{self.format_indian_currency(rate_stats['validation_range_upper'])}"
        else:
            rate_message = f"Outside range: ₹{self.format_indian_currency(rate_stats['validation_range_lower'])} - ₹{self.format_indian_currency(rate_stats['validation_range_upper'])}"
        
        return {
            'cwi_valid': cwi_valid,
            'rate_valid': rate_valid,
            'standard_group': standard_group,
            'cwi_message': cwi_message,
            'rate_message': rate_message,
            'cwi_stats': cwi_stats,
            'rate_stats': rate_stats
        }
   
    def load_model(self, model_path):
        """Load trained model from file"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.sqfeet_ranges = model_data['sqfeet_ranges']
            self.vectorizers = model_data['vectorizers']
            self.particular_vectors = model_data['particular_vectors']
            self.particular_names = model_data['particular_names']
            self.training_stats = model_data['training_stats']
            self.known_groups = model_data.get('known_groups', self.known_groups)
            self.all_groups = model_data.get('all_groups', set())
            self.new_groups_found = model_data.get('new_groups_found', set())
            self.unknown_particulars = model_data.get('unknown_particulars', set())
            
            print(f"✓ Grouped model loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading grouped model: {str(e)}")
            return False

# --- Flask Application Setup ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'super-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_PATHS = {
    'hdfc_qty': "trained_model_sqfeet_zero_aware.pkl",
    'hdfc_amount': "trained_model_sqfeet_zero_aware_logic_amount_HDFC.pkl",  # ADD THIS
    'hdfc_rate': "trained_model_sqfeet_zero_aware_logic_rate_HDFC.pkl",
    'hdfc_grouped': "trained_model_sqfeet_zero_aware_logic_grouped_HDFC.pkl",
    'icici_qty': "trained_model_sqfeet_weighted_avg_ICICI.pkl",
    'icici_amount': "trained_model_sqfeet_zero_aware_logic_amount_ICICI.pkl",
    'icici_rate': "trained_model_sqfeet_zero_aware_logic_rate_ICICI.pkl",
    'icici_grouped': "trained_model_sqfeet_zero_aware_logic_grouped_ICICI.pkl",  # ADD THIS LINE
    'icici_rate_east': "trained_model_sqfeet_zero_aware_logic_rate_EAST.pkl",
    'icici_rate_west': "trained_model_sqfeet_zero_aware_logic_rate_WEST.pkl",
    'icici_rate_north': "trained_model_sqfeet_zero_aware_logic_rate_NORTH.pkl",
    'icici_rate_south': "trained_model_sqfeet_zero_aware_logic_rate_SOUTH.pkl",
    'icici_qty_east': "trained_model_sqfeet_zero_aware_logic_qty_EAST.pkl",  # You'll train these later
    'icici_qty_west': "trained_model_sqfeet_zero_aware_logic_qty_WEST.pkl",
    'icici_qty_north': "trained_model_sqfeet_zero_aware_logic_qty_NORTH.pkl",
    'icici_qty_south': "trained_model_sqfeet_zero_aware_logic_qty_SOUTH.pkl"

}

VALIDATORS = {}
GROUPED_VALIDATORS = {}  # NEW: Separate dict for grouped validators

for client, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path):
        if 'grouped' in client:
            validator = CWIGroupedValidator()  # Use grouped validator class
            validator.load_model(model_path)
            GROUPED_VALIDATORS[client] = validator
            print(f"✅ {client.upper()} Grouped Validator loaded successfully.")
        else:
            validator = CWIValidatorTester()
            validator.load_model(model_path)
            VALIDATORS[client] = validator
            print(f"✅ {client.upper()} Validator Model loaded successfully.")
    else:
        print(f"❌ WARNING: {client.upper()} model file not found at '{model_path}'!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_thresholds(message):
    if pd.isna(message): return "NA", "NA"
    # NEW: Updated regex to handle both integer and decimal ranges
    range_match = re.search(r'Range.*?:\s*([\d.]+)\s*-\s*([\d.]+)', str(message))
    if range_match: 
        val1 = range_match.group(1)
        val2 = range_match.group(2)
        # NEW: Return as int if no decimal point, else float
        return (int(val1) if '.' not in val1 else float(val1), 
                int(val2) if '.' not in val2 else float(val2))
    if "Zero quantity allowed" in str(message): return 0, 0
    return "NA", "NA"

def highlight_invalid(row):
    color = '#FFC7CE'
    style = f'background-color: {color}'
    styles = [''] * len(row)
    if not row['Is_Valid']:
        # Check which column exists: 'Quantity', 'Amount', or 'Rate'
        if 'Amount' in row.index:
            value_col = 'Amount'
        elif 'Rate' in row.index:  # NEW
            value_col = 'Rate'
        else:
            value_col = 'Quantity'
        styles[row.index.get_loc(value_col)] = style
        styles[row.index.get_loc('Is_Valid')] = style
    return styles
def validate_grouped_file(validator, filepath):
    """Validate grouped per sq.feet data from Summary Sheet"""
    try:
        df = pd.read_excel(filepath, sheet_name='Summary Sheet')
        
        # Find CWI column
        cwi_col = None
        for col in df.columns:
            if 'cwi' in str(col).lower():
                cwi_col = col
                break
        
        if cwi_col is None:
            return None, "Could not find column containing 'CWI' in Summary Sheet"
        
        # Last column is Rate per sq.feet
        rate_col = df.columns[-1]
        
        # Find total row to stop processing
        total_idx = None
        for idx, row in df.iterrows():
            first_val = str(row.iloc[0]).strip().lower() if not pd.isna(row.iloc[0]) else ''
            second_val = str(row.iloc[1]).strip().lower() if not pd.isna(row.iloc[1]) else ''
            if first_val in ['total', 'grand total'] or second_val in ['total', 'grand total']:
                total_idx = idx
                break
        
        if total_idx is not None:
            df = df.iloc[:total_idx + 1]
        
        # Extract sq feet range from filename
        sqfeet_range = validator.extract_sqfeet_from_filename(os.path.basename(filepath))
        if not sqfeet_range:
            return None, "Could not determine sq feet range from filename"
        
        results = []
        for idx, row in df.iterrows():
            sr_no = row.iloc[0]
            particular = row.iloc[1]
            
            if pd.isna(particular) or str(particular).strip() == '':
                continue
            
            # Extract amounts
            cwi_amount = validator.extract_amount(row[cwi_col])
            rate = validator.extract_amount(row[rate_col])
            
            # Validate
            validation_result = validator.validate_grouped_data(
                particular, cwi_amount, rate, sqfeet_range
            )
            
            # ✅ NEW: Extract weighted averages from stats
            cwi_stats = validation_result.get('cwi_stats', {})
            rate_stats = validation_result.get('rate_stats', {})
            
            result_dict = {
                'Sr.No': sr_no,
                'Particulars': str(particular),
                'As_Per_CWI_Amount': cwi_amount,
                'Rate_per_sqfeet': rate,
                'Sq_Feet_Range': sqfeet_range,
                'Is_Valid': validation_result['cwi_valid'] and validation_result['rate_valid'],
                'CWI_Validation': validation_result.get('cwi_message', ''),
                'Rate_Validation': validation_result.get('rate_message', ''),
                # ✅ NEW: Add weighted averages
                'CWI_Weighted_Min': cwi_stats.get('weighted_avg_min', 'NA'),
                'CWI_Weighted_Max': cwi_stats.get('weighted_avg_max', 'NA'),
                'Rate_Weighted_Min': rate_stats.get('weighted_avg_min', 'NA'),
                'Rate_Weighted_Max': rate_stats.get('weighted_avg_max', 'NA')
            }
            results.append(result_dict)
        
        return results, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error processing grouped file: {str(e)}"
def process_comparison(hdfc_filepath, icici_filepath):
    """Process HDFC vs ICICI comparison for grouped validation"""
    try:
        print("\n" + "="*60)
        print("🔄 COMPARISON MODE ACTIVATED")
        print("="*60)
        
        # Load both grouped validators
        hdfc_validator = GROUPED_VALIDATORS.get('hdfc_grouped')
        icici_validator = GROUPED_VALIDATORS.get('icici_grouped')
        
        if not hdfc_validator or not icici_validator:
            print("❌ Comparison models not available")
            return None, "Comparison models not available. Please ensure both HDFC and ICICI grouped models are loaded."
        
        print("✅ Both validators loaded successfully")
        
        # Validate both files
        print("🔍 Validating HDFC file...")
        hdfc_results, hdfc_error = validate_grouped_file(hdfc_validator, hdfc_filepath)
        
        print("🔍 Validating ICICI file...")
        icici_results, icici_error = validate_grouped_file(icici_validator, icici_filepath)
        
        if hdfc_error:
            print(f"❌ HDFC validation error: {hdfc_error}")
            return None, f"HDFC validation error: {hdfc_error}"
        
        if icici_error:
            print(f"❌ ICICI validation error: {icici_error}")
            return None, f"ICICI validation error: {icici_error}"
        
        print(f"✅ HDFC: {len(hdfc_results)} records validated")
        print(f"✅ ICICI: {len(icici_results)} records validated")
        
        # Convert to DataFrames
        hdfc_df = pd.DataFrame(hdfc_results)
        icici_df = pd.DataFrame(icici_results)
        
        # Format currency values for HDFC
        print("💰 Formatting HDFC currency values...")
        hdfc_df['As_Per_CWI_Amount'] = hdfc_df['As_Per_CWI_Amount'].apply(
            lambda x: hdfc_validator.format_indian_currency(x)
        )
        hdfc_df['Rate_per_sqfeet'] = hdfc_df['Rate_per_sqfeet'].apply(
            lambda x: hdfc_validator.format_indian_currency(x)
        )
        hdfc_df['CWI_Weighted_Min'] = hdfc_df['CWI_Weighted_Min'].apply(
            lambda x: hdfc_validator.format_indian_currency(x) if x != 'NA' and not pd.isna(x) else 'NA'
        )
        hdfc_df['CWI_Weighted_Max'] = hdfc_df['CWI_Weighted_Max'].apply(
            lambda x: hdfc_validator.format_indian_currency(x) if x != 'NA' and not pd.isna(x) else 'NA'
        )
        hdfc_df['Rate_Weighted_Min'] = hdfc_df['Rate_Weighted_Min'].apply(
            lambda x: hdfc_validator.format_indian_currency(x) if x != 'NA' and not pd.isna(x) else 'NA'
        )
        hdfc_df['Rate_Weighted_Max'] = hdfc_df['Rate_Weighted_Max'].apply(
            lambda x: hdfc_validator.format_indian_currency(x) if x != 'NA' and not pd.isna(x) else 'NA'
        )
        
        # Format currency values for ICICI
        print("💰 Formatting ICICI currency values...")
        icici_df['As_Per_CWI_Amount'] = icici_df['As_Per_CWI_Amount'].apply(
            lambda x: icici_validator.format_indian_currency(x)
        )
        icici_df['Rate_per_sqfeet'] = icici_df['Rate_per_sqfeet'].apply(
            lambda x: icici_validator.format_indian_currency(x)
        )
        icici_df['CWI_Weighted_Min'] = icici_df['CWI_Weighted_Min'].apply(
            lambda x: icici_validator.format_indian_currency(x) if x != 'NA' and not pd.isna(x) else 'NA'
        )
        icici_df['CWI_Weighted_Max'] = icici_df['CWI_Weighted_Max'].apply(
            lambda x: icici_validator.format_indian_currency(x) if x != 'NA' and not pd.isna(x) else 'NA'
        )
        icici_df['Rate_Weighted_Min'] = icici_df['Rate_Weighted_Min'].apply(
            lambda x: icici_validator.format_indian_currency(x) if x != 'NA' and not pd.isna(x) else 'NA'
        )
        icici_df['Rate_Weighted_Max'] = icici_df['Rate_Weighted_Max'].apply(
            lambda x: icici_validator.format_indian_currency(x) if x != 'NA' and not pd.isna(x) else 'NA'
        )
        
        # Format Is_Valid column
        hdfc_df['Is_Valid'] = hdfc_df['Is_Valid'].apply(lambda x: '✅ Valid' if x else '❌ Invalid')
        icici_df['Is_Valid'] = icici_df['Is_Valid'].apply(lambda x: '✅ Valid' if x else '❌ Invalid')
        
        # Save comparison reports
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        hdfc_report = f'hdfc_comparison_{timestamp}.xlsx'
        icici_report = f'icici_comparison_{timestamp}.xlsx'
        
        hdfc_df.to_excel(os.path.join(app.config['OUTPUT_FOLDER'], hdfc_report), index=False)
        icici_df.to_excel(os.path.join(app.config['OUTPUT_FOLDER'], icici_report), index=False)
        
        print(f"✅ Reports saved: {hdfc_report}, {icici_report}")
        print("="*60 + "\n")
        
        # Convert to HTML for display
        hdfc_html = hdfc_df.to_html(classes='table table-striped', index=False, escape=False)
        icici_html = icici_df.to_html(classes='table table-striped', index=False, escape=False)
        
        return {
            'hdfc_table': hdfc_html,
            'icici_table': icici_html,
            'hdfc_report': hdfc_report,
            'icici_report': icici_report
        }, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Comparison error: {str(e)}")
        return None, f'Comparison error: {str(e)}'

def process_qar_analysis(hdfc_filepath, icici_filepath):
    """Process QAR (Qty/Amount/Rate) Analysis between ICICI and HDFC"""
    try:
        print("\n" + "="*60)
        print("🔄 QAR ANALYSIS MODE ACTIVATED")
        print("="*60)
        
        # ✅ FIX: Load ALL three validators for each bank
        hdfc_qty_validator = VALIDATORS.get('hdfc_qty')
        hdfc_amount_validator = VALIDATORS.get('hdfc_amount')
        hdfc_rate_validator = VALIDATORS.get('hdfc_rate')
        
        icici_qty_validator = VALIDATORS.get('icici_qty')
        icici_amount_validator = VALIDATORS.get('icici_amount')
        icici_rate_validator = VALIDATORS.get('icici_rate')
        
        if not all([hdfc_qty_validator, hdfc_amount_validator, hdfc_rate_validator,
                    icici_qty_validator, icici_amount_validator, icici_rate_validator]):
            return None, "QAR analysis models not available. Please ensure qty, amount, and rate models are loaded for both banks."
        
        print("✅ All QAR validators loaded successfully")
        
        # Read both Excel files
        hdfc_df = pd.read_excel(hdfc_filepath, sheet_name='Extracted Data')
        icici_df = pd.read_excel(icici_filepath, sheet_name='Extracted Data')
        
        # Extract sq feet ranges
        hdfc_sqft_qty = hdfc_qty_validator.extract_sqfeet_from_filename(os.path.basename(hdfc_filepath), mode='quantity')
        hdfc_sqft_amt = hdfc_amount_validator.extract_sqfeet_from_filename(os.path.basename(hdfc_filepath), mode='amount')
        hdfc_sqft_rate = hdfc_rate_validator.extract_sqfeet_from_filename(os.path.basename(hdfc_filepath), mode='rate')
        
        icici_sqft_qty = icici_qty_validator.extract_sqfeet_from_filename(os.path.basename(icici_filepath), mode='quantity')
        icici_sqft_amt = icici_amount_validator.extract_sqfeet_from_filename(os.path.basename(icici_filepath), mode='amount')
        icici_sqft_rate = icici_rate_validator.extract_sqfeet_from_filename(os.path.basename(icici_filepath), mode='rate')
        
        # Find vitrified tiles 800x800 particular
        # ✅ NEW: Find all 10 matching particulars
        particulars_to_find = [
            ('Vitrified Tiles 800x800', find_vitrified_800x800),
            ('Siporex 9 inch', find_siporex_9inch),
            ('R.C.C Cement Concrete', find_rcc_cement_concrete),
            ('Aluminum Framework BWP', find_aluminum_framework_bwp),
            ('Ceramic Tiles 300x600', find_ceramic_tiles_300x600),
            ('Aluminum Composite Panel', find_aluminum_composite_panel),
            ('12mm Plaster 1:4', find_12mm_plaster_1_4),
            ('Service Counter Table', find_service_counter_table),
            ('Full Height Partition BWR', find_full_height_partition_bwr),
            ('Gypsum False Ceiling', find_gypsum_false_ceiling),
            ('Plaster of Paris Punning', find_plaster_of_paris_punning),
            ('Acrylic Emulsion Paint', find_acrylic_emulsion_paint),  # ✅ NEW
            ('Modular Ceiling', find_modular_ceiling) 
        ]
        
        # Around line 1100-1200, replace this section:

        comparison_results = []
        icici_only_results = []

        for particular_name, find_function in particulars_to_find:
            print(f"🔍 Searching for: {particular_name}")

            icici_row = find_function(icici_df, 'icici')
            hdfc_row = find_function(hdfc_df, 'hdfc')

            if icici_row is None:
                print(f"  ⚠️  Not found in ICICI: {particular_name}")
                continue

            print(f"  ✅ Found in ICICI: {icici_row['Particulars']}")

            # Part 1: ICICI vs HDFC Comparison
            if hdfc_row is not None:
                print(f"  ✅ Found in HDFC: {hdfc_row['Particulars']}")

                comparison_result = compare_icici_with_other(
                    icici_row, hdfc_row, 
                    icici_qty_validator, icici_amount_validator, icici_rate_validator,
                    hdfc_qty_validator, hdfc_amount_validator, hdfc_rate_validator,
                    icici_sqft_qty, icici_sqft_amt, icici_sqft_rate,
                    hdfc_sqft_qty, hdfc_sqft_amt, hdfc_sqft_rate
                )
                comparison_result['Particular_Category'] = particular_name
                comparison_results.append(comparison_result)
            else:
                print(f"  ⚠️  Not found in HDFC: {particular_name}")

            # Part 2: ICICI Bank Only Analysis - THIS MUST RUN FOR EVERY ICICI ROW
            icici_only_result = analyze_icici_only(
                icici_row, 
                icici_qty_validator, icici_amount_validator, icici_rate_validator,
                icici_sqft_qty, icici_sqft_amt, icici_sqft_rate
            )
            icici_only_result['Particular_Category'] = particular_name
            icici_only_results.append(icici_only_result)  # ✅ ADD TO LIST, NOT OVERWRITE

        # Final check
        if not comparison_results and not icici_only_results:
            return None, "Could not find any matching particulars in the files"

        return {
            'comparison_results': comparison_results,
            'icici_only_results': icici_only_results,  # ✅ CHANGE FROM icici_only_result
            'hdfc_file': os.path.basename(hdfc_filepath),
            'icici_file': os.path.basename(icici_filepath)
        }, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"QAR Analysis error: {str(e)}"

def find_vitrified_800x800(df, bank_type):
    """Find the vitrified tiles 800x800 particular from dataframe"""
    # Map column names
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    # Search for 800x800 vitrified tiles
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        # Check for 800x800 pattern
        has_size = bool(re.search(r'800\s*[x*×]\s*800', particular))
        has_vitrified = 'vitrified' in particular and 'tiles' in particular
        
        if has_size and has_vitrified:
            return {
                'Sr.No': idx + 1,
                'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                'Qty': row.get(col_mapping.get('Qty', ''), 0),
                'Amount': row.get(col_mapping.get('Amount', ''), 0),
                'Rate': row.get(col_mapping.get('Rate', ''), 0),
                'Unit': row.get(col_mapping.get('Unit', ''), '')
            }
    
    return None

def find_siporex_9inch(df, bank_type):
    """Find Siporex 9 inch particular from dataframe"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # Check for 'siporex' AND ('9inch' OR '9 inch')
            has_siporex = 'siporex' in particular
            has_9inch = bool(re.search(r'9\s*inch', particular))
            
            if has_siporex and has_9inch:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for '150mm' AND 'siporex'
            has_150mm = bool(re.search(r'150\s*mm', particular))
            has_siporex = 'siporex' in particular
            
            if has_150mm and has_siporex:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None

def find_rcc_cement_concrete(df, bank_type):
    """Find R.C.C Work with providing and casting"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # Check for 'r.c.c' or 'rcc' AND 'work' AND 'providing' AND 'casting'
            has_rcc = bool(re.search(r'r\.?\s*c\.?\s*c', particular))
            has_work = 'work' in particular
            has_providing = 'providing' in particular
            has_casting = 'casting' in particular
            
            if has_rcc and has_work and has_providing and has_casting:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for 'r.c.c' or 'rcc' AND 'work' AND 'providing' AND ('casting' OR '&')
            has_rcc = bool(re.search(r'r\.?\s*c\.?\s*c', particular))
            has_work = 'work' in particular
            has_providing = 'providing' in particular
            # Accept both 'casting' and '&' (ampersand often used as "and")
            has_casting_or_ampersand = 'casting' in particular or '&' in particular
            
            if has_rcc and has_work and has_providing and has_casting_or_ampersand:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None

def find_aluminum_framework_bwp(df, bank_type):
    """Find Aluminum framework with BWP plywood"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # Check for 'wall' or 'column', 'cladding', '2 x 1.5', 'aluminum'/'aluminium'
            has_wall_column = 'wall' in particular or 'column' in particular
            has_cladding = 'cladding' in particular
            has_2x15 = bool(re.search(r'2\s*[x×]\s*1\.?5', particular))
            has_aluminum = 'aluminum' in particular or 'aluminium' in particular
            
            if has_wall_column and has_cladding and has_2x15 and has_aluminum:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for 'aluminium'/'aluminum', 'frame', 'bwp', 'plywood'
            has_aluminum = 'aluminum' in particular or 'aluminium' in particular
            has_frame = 'frame' in particular
            has_bwp = 'bwp' in particular
            has_plywood = 'plywood' in particular
            
            if has_aluminum and has_frame and has_bwp and has_plywood:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None

def find_ceramic_tiles_300x600(df, bank_type):
    """Find ceramic tiles 300x600mm"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # Check for 'ceramic tiles' AND '300x600' or '300 x 600'
            has_ceramic = 'ceramic' in particular and 'tiles' in particular
            has_300x600 = bool(re.search(r'300\s*[x×]\s*600', particular))
            
            if has_ceramic and has_300x600:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for '600mm x 600mm' or '600 x 600' AND 'vitrified'
            has_600x600 = bool(re.search(r'600\s*(?:mm)?\s*[x×]\s*600', particular))
            has_vitrified = 'vitrified' in particular
            
            if has_600x600 and has_vitrified:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None

def find_aluminum_composite_panel(df, bank_type):
    """Find Aluminum Composite Panel"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # ✅ NEW: Check for 'exterior grade' AND '3mm' AND 'thick' AND 'aluminum' AND 'composite' AND 'panel'
            # ❌ IGNORE if contains 'silver grey' or 'silver' AND 'grey'
            has_exterior = 'exterior' in particular and 'grade' in particular
            has_3mm = '3mm' in particular or '3 mm' in particular
            has_thick = 'thick' in particular
            has_aluminum = 'aluminum' in particular or 'aluminium' in particular
            has_composite = 'composite' in particular
            has_panel = 'panel' in particular
            
            # Check if it contains "silver grey" or "silver" + "grey"
            has_silver_grey = ('silver' in particular and 'grey' in particular) or 'silver grey' in particular or 'silvergrey' in particular
            
            # Match if all required keywords present AND does NOT have silver grey
            if has_exterior and has_3mm and has_thick and has_aluminum and has_composite and has_panel and not has_silver_grey:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for 'providing' AND 'fixing' AND 'tricolour'
            has_providing = 'providing' in particular
            has_fixing = 'fixing' in particular
            has_tricolour = 'tricolour' in particular or 'tricolor' in particular
            
            if has_providing and has_fixing and has_tricolour:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None

def find_12mm_plaster_1_4(df, bank_type):
    """Find 12mm thick plaster 1:4"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # Check for '12mm' OR '12 mm' AND 'plaster' AND '1:4'
            has_12mm = bool(re.search(r'12\s*mm', particular))
            has_plaster = 'plaster' in particular
            has_1_4 = '1:4' in particular or '1 : 4' in particular
            
            if has_12mm and has_plaster and has_1_4:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for '1:4' AND 'cement plaster' AND '12mm' OR '12 mm'
            has_1_4 = '1:4' in particular or '1 : 4' in particular
            has_cement_plaster = 'cement' in particular and 'plaster' in particular
            has_12mm = bool(re.search(r'12\s*mm', particular))
            
            if has_1_4 and has_cement_plaster and has_12mm:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None

def find_service_counter_table(df, bank_type):
    """Find modular service counter table"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # Check for 'modular service counter table' AND '0.75 inch'
            has_modular = 'modular' in particular
            has_service_counter = 'service' in particular and 'counter' in particular
            has_table = 'table' in particular
            has_075 = bool(re.search(r'0\.75\s*inch', particular))
            
            if has_modular and has_service_counter and has_table and has_075:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for "branch manager's table" OR "branch manager table" AND '1350 x 675' or '1350x675'
            has_branch_manager = 'branch' in particular and 'manager' in particular and 'table' in particular
            has_1350x675 = bool(re.search(r'1350\s*[x×]\s*675', particular))
            
            if has_branch_manager and has_1350x675:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None

def find_full_height_partition_bwr(df, bank_type):
    """Find full height partitions with BWR plywood"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # Check for 'full height partitions' AND '2 x 1.5'
            has_full_height = 'full' in particular and 'height' in particular
            has_partitions = 'partition' in particular
            has_2x15 = bool(re.search(r'2\s*[x×]\s*1\.?5', particular))
            
            if has_full_height and has_partitions and has_2x15:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for 'full height' AND 'solid partition' AND 'bwr plywood'
            has_full_height = 'full' in particular and 'height' in particular
            has_solid_partition = 'solid' in particular and 'partition' in particular
            has_bwr = 'bwr' in particular
            has_plywood = 'plywood' in particular
            
            if has_full_height and has_solid_partition and has_bwr and has_plywood:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None

def find_gypsum_false_ceiling(df, bank_type):
    """Find Gypsum False Ceiling"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # Check for 'gypsum' AND 'false ceiling'
            has_gypsum = 'gypsum' in particular
            has_false_ceiling = 'false' in particular and 'ceiling' in particular
            
            if has_gypsum and has_false_ceiling:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for 'gypsum' AND 'board' AND 'ceiling'
            has_gypsum = 'gypsum' in particular
            has_board = 'board' in particular
            has_ceiling = 'ceiling' in particular
            
            if has_gypsum and has_board and has_ceiling:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None

def find_plaster_of_paris_punning(df, bank_type):
    """Find Plaster of Paris punning"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        # Check for 'plaster of paris' AND 'punning'
        has_pop = 'plaster' in particular and 'paris' in particular
        has_punning = 'punning' in particular
        
        if has_pop and has_punning:
            return {
                'Sr.No': idx + 1,
                'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                'Qty': row.get(col_mapping.get('Qty', ''), 0),
                'Amount': row.get(col_mapping.get('Amount', ''), 0),
                'Rate': row.get(col_mapping.get('Rate', ''), 0),
                'Unit': row.get(col_mapping.get('Unit', ''), '')
            }
    
    return None

def find_acrylic_emulsion_paint(df, bank_type):
    """Find acrylic emulsion paint particular from dataframe"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # Check for 'acrylic' AND 'emulsion' AND 'paint'
            has_acrylic = 'acrylic' in particular
            has_emulsion = 'emulsion' in particular
            has_paint = 'paint' in particular
            
            if has_acrylic and has_emulsion and has_paint:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for 'applying' AND 'acrylic' AND 'emulsion' AND 'paint'
            has_applying = 'applying' in particular
            has_acrylic = 'acrylic' in particular
            has_emulsion = 'emulsion' in particular
            has_paint = 'paint' in particular
            
            if has_applying and has_acrylic and has_emulsion and has_paint:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None

def find_modular_ceiling(df, bank_type):
    """Find modular ceiling particular from dataframe"""
    col_mapping = {}
    for col in df.columns:
        clean_col = str(col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if 'particular' in clean_col:
            col_mapping['Particulars'] = col
        elif 'qty' in clean_col and 'cwi' in clean_col:
            col_mapping['Qty'] = col
        elif 'amount' in clean_col and 'cwi' in clean_col:
            col_mapping['Amount'] = col
        elif 'rate' in clean_col:
            col_mapping['Rate'] = col
        elif 'unit' in clean_col:
            col_mapping['Unit'] = col
    
    for idx, row in df.iterrows():
        particular = str(row.get(col_mapping.get('Particulars', ''), '')).lower()
        
        if bank_type == 'icici':
            # Check for 'modular' AND 'ceiling' (case-insensitive)
            has_modular = 'modular' in particular
            has_ceiling = 'ceiling' in particular
            
            if has_modular and has_ceiling:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
        else:  # hdfc
            # Check for 'grid' AND 'modular' AND 'false' AND 'ceiling'
            has_grid = 'grid' in particular
            has_modular = 'modular' in particular
            has_false = 'false' in particular
            has_ceiling = 'ceiling' in particular
            
            if has_grid and has_modular and has_false and has_ceiling:
                return {
                    'Sr.No': idx + 1,
                    'Particulars': str(row.get(col_mapping.get('Particulars', ''), '')),
                    'Qty': row.get(col_mapping.get('Qty', ''), 0),
                    'Amount': row.get(col_mapping.get('Amount', ''), 0),
                    'Rate': row.get(col_mapping.get('Rate', ''), 0),
                    'Unit': row.get(col_mapping.get('Unit', ''), '')
                }
    
    return None


def compare_icici_with_other(icici_row, hdfc_row, 
                             icici_qty_val, icici_amt_val, icici_rate_val,
                             hdfc_qty_val, hdfc_amt_val, hdfc_rate_val,
                             icici_sqft_qty, icici_sqft_amt, icici_sqft_rate,
                             hdfc_sqft_qty, hdfc_sqft_amt, hdfc_sqft_rate):
    """Compare ICICI with HDFC for the particular using separate validators for qty, amount, rate"""
    
    result = {
        'Sr.No': 1,
        'ICICI_Particulars': icici_row['Particulars'],
        'ICICI_Qty': icici_qty_val.extract_quantity(icici_row['Qty']),
        'ICICI_Amount': icici_amt_val.extract_amount(icici_row['Amount']),
        'ICICI_Rate': icici_rate_val.extract_amount(icici_row['Rate']),
        'Other_Bank_Particulars': hdfc_row['Particulars']
    }
    
    # Get HDFC cleaned particular
    hdfc_cleaned = hdfc_qty_val.clean_text(hdfc_row['Particulars'])
    
    # ===== QUANTITY VALIDATION (using QTY model) =====
    if hdfc_sqft_qty in hdfc_qty_val.sqfeet_ranges:
        range_data = hdfc_qty_val.sqfeet_ranges[hdfc_sqft_qty]
        range_key = 'quantity_ranges' if 'quantity_ranges' in range_data else 'rate_ranges' if 'rate_ranges' in range_data else 'amount_ranges'
        
        if range_key in range_data and hdfc_cleaned in range_data[range_key]:
            qty_stats = range_data[range_key][hdfc_cleaned]
            weighted_min_qty = qty_stats.get('weighted_avg_min', qty_stats.get('overall_mean', 0))
            weighted_max_qty = qty_stats.get('weighted_avg_max', qty_stats.get('overall_mean', 0))
            
            result['Is_Valid_Qty_Other'] = weighted_min_qty <= result['ICICI_Qty'] <= weighted_max_qty
            result['Threshold_Range_Qty'] = f"{qty_stats.get('overall_min', 0):.2f} - {qty_stats.get('overall_max', 0):.2f}"
            result['Weighted_Avg_Range_Qty'] = f"{weighted_min_qty:.2f} - {weighted_max_qty:.2f}"
        else:
            result['Is_Valid_Qty_Other'] = 'N/A'
            result['Threshold_Range_Qty'] = 'N/A'
            result['Weighted_Avg_Range_Qty'] = 'N/A'
    else:
        result['Is_Valid_Qty_Other'] = 'N/A'
        result['Threshold_Range_Qty'] = 'N/A'
        result['Weighted_Avg_Range_Qty'] = 'N/A'
    
    # ===== AMOUNT VALIDATION (using AMOUNT model) =====
    if hdfc_sqft_amt in hdfc_amt_val.sqfeet_ranges:
        range_data = hdfc_amt_val.sqfeet_ranges[hdfc_sqft_amt]
        range_key = 'amount_ranges' if 'amount_ranges' in range_data else 'rate_ranges' if 'rate_ranges' in range_data else 'quantity_ranges'
        
        if range_key in range_data and hdfc_cleaned in range_data[range_key]:
            amt_stats = range_data[range_key][hdfc_cleaned]
            weighted_min_amt = amt_stats.get('weighted_avg_min', amt_stats.get('overall_mean', 0))
            weighted_max_amt = amt_stats.get('weighted_avg_max', amt_stats.get('overall_mean', 0))
            
            result['Is_Valid_Amount_Other'] = weighted_min_amt <= result['ICICI_Amount'] <= weighted_max_amt
            result['Threshold_Range_Amount'] = f"₹{hdfc_amt_val.format_indian_currency(amt_stats.get('overall_min', 0))} - ₹{hdfc_amt_val.format_indian_currency(amt_stats.get('overall_max', 0))}"
            result['Weighted_Avg_Range_Amount'] = f"₹{hdfc_amt_val.format_indian_currency(weighted_min_amt)} - ₹{hdfc_amt_val.format_indian_currency(weighted_max_amt)}"
        else:
            result['Is_Valid_Amount_Other'] = 'N/A'
            result['Threshold_Range_Amount'] = 'N/A'
            result['Weighted_Avg_Range_Amount'] = 'N/A'
    else:
        result['Is_Valid_Amount_Other'] = 'N/A'
        result['Threshold_Range_Amount'] = 'N/A'
        result['Weighted_Avg_Range_Amount'] = 'N/A'
    
    # ===== RATE VALIDATION (using RATE model) =====
    if hdfc_sqft_rate in hdfc_rate_val.sqfeet_ranges:
        range_data = hdfc_rate_val.sqfeet_ranges[hdfc_sqft_rate]
        range_key = 'rate_ranges' if 'rate_ranges' in range_data else 'amount_ranges' if 'amount_ranges' in range_data else 'quantity_ranges'
        
        if range_key in range_data and hdfc_cleaned in range_data[range_key]:
            rate_stats = range_data[range_key][hdfc_cleaned]
            weighted_min_rate = rate_stats.get('weighted_avg_min', rate_stats.get('overall_mean', 0))
            weighted_max_rate = rate_stats.get('weighted_avg_max', rate_stats.get('overall_mean', 0))
            
            result['Is_Valid_Rate_Other'] = weighted_min_rate <= result['ICICI_Rate'] <= weighted_max_rate
            result['Threshold_Range_Rate'] = f"₹{hdfc_rate_val.format_indian_currency(rate_stats.get('overall_min', 0))} - ₹{hdfc_rate_val.format_indian_currency(rate_stats.get('overall_max', 0))}"
            result['Weighted_Avg_Range_Rate'] = f"₹{hdfc_rate_val.format_indian_currency(weighted_min_rate)} - ₹{hdfc_rate_val.format_indian_currency(weighted_max_rate)}"
        else:
            result['Is_Valid_Rate_Other'] = 'N/A'
            result['Threshold_Range_Rate'] = 'N/A'
            result['Weighted_Avg_Range_Rate'] = 'N/A'
    else:
        result['Is_Valid_Rate_Other'] = 'N/A'
        result['Threshold_Range_Rate'] = 'N/A'
        result['Weighted_Avg_Range_Rate'] = 'N/A'

    # Create validation messages
    if result.get('Is_Valid_Qty_Other') == True:
        result['Validation_Message_Qty'] = f"✅ Within range: {result.get('Weighted_Avg_Range_Qty', 'N/A')}"
    elif result.get('Is_Valid_Qty_Other') == False:
        result['Validation_Message_Qty'] = f"❌ Outside range: {result.get('Weighted_Avg_Range_Qty', 'N/A')}"
    else:
        result['Validation_Message_Qty'] = "N/A - No validation data"
    
    if result.get('Is_Valid_Amount_Other') == True:
        result['Validation_Message_Amount'] = f"✅ Within range: {result.get('Weighted_Avg_Range_Amount', 'N/A')}"
    elif result.get('Is_Valid_Amount_Other') == False:
        result['Validation_Message_Amount'] = f"❌ Outside range: {result.get('Weighted_Avg_Range_Amount', 'N/A')}"
    else:
        result['Validation_Message_Amount'] = "N/A - No validation data"
    
    if result.get('Is_Valid_Rate_Other') == True:
        result['Validation_Message_Rate'] = f"✅ Within range: {result.get('Weighted_Avg_Range_Rate', 'N/A')}"
    elif result.get('Is_Valid_Rate_Other') == False:
        result['Validation_Message_Rate'] = f"❌ Outside range: {result.get('Weighted_Avg_Range_Rate', 'N/A')}"
    else:
        result['Validation_Message_Rate'] = "N/A - No validation data"
    
    return result


def analyze_icici_only(icici_row, 
                       icici_qty_val, icici_amt_val, icici_rate_val,
                       icici_sqft_qty, icici_sqft_amt, icici_sqft_rate):
    """Analyze ICICI data against its own models (qty, amount, rate separately)"""
    
    result = {
        'Sr.No': 1,
        'Particulars': icici_row['Particulars'],
        'Qty': icici_qty_val.extract_quantity(icici_row['Qty']),
        'Amount': icici_amt_val.extract_amount(icici_row['Amount']),
        'Rate': icici_rate_val.extract_amount(icici_row['Rate'])
    }
    
    icici_cleaned = icici_qty_val.clean_text(icici_row['Particulars'])
    
    # ===== QUANTITY (using QTY model) =====
    if icici_sqft_qty in icici_qty_val.sqfeet_ranges:
        range_data = icici_qty_val.sqfeet_ranges[icici_sqft_qty]
        range_key = 'quantity_ranges' if 'quantity_ranges' in range_data else 'rate_ranges' if 'rate_ranges' in range_data else 'amount_ranges'
        
        if range_key in range_data and icici_cleaned in range_data[range_key]:
            qty_stats = range_data[range_key][icici_cleaned]
            weighted_min_qty = qty_stats.get('weighted_avg_min', qty_stats.get('overall_mean', 0))
            weighted_max_qty = qty_stats.get('weighted_avg_max', qty_stats.get('overall_mean', 0))
            
            result['Is_Valid_Qty'] = weighted_min_qty <= result['Qty'] <= weighted_max_qty
            result['Threshold_Range_Qty'] = f"{qty_stats.get('overall_min', 0):.2f} - {qty_stats.get('overall_max', 0):.2f}"
            result['Weighted_Avg_Range_Qty'] = f"{weighted_min_qty:.2f} - {weighted_max_qty:.2f}"
        else:
            result['Is_Valid_Qty'] = 'N/A'
            result['Threshold_Range_Qty'] = 'N/A'
            result['Weighted_Avg_Range_Qty'] = 'N/A'
    else:
        result['Is_Valid_Qty'] = 'N/A'
        result['Threshold_Range_Qty'] = 'N/A'
        result['Weighted_Avg_Range_Qty'] = 'N/A'
    
    # ===== AMOUNT (using AMOUNT model) =====
    if icici_sqft_amt in icici_amt_val.sqfeet_ranges:
        range_data = icici_amt_val.sqfeet_ranges[icici_sqft_amt]
        range_key = 'amount_ranges' if 'amount_ranges' in range_data else 'rate_ranges' if 'rate_ranges' in range_data else 'quantity_ranges'
        
        if range_key in range_data and icici_cleaned in range_data[range_key]:
            amt_stats = range_data[range_key][icici_cleaned]
            weighted_min_amt = amt_stats.get('weighted_avg_min', amt_stats.get('overall_mean', 0))
            weighted_max_amt = amt_stats.get('weighted_avg_max', amt_stats.get('overall_mean', 0))
            
            result['Is_Valid_Amount'] = weighted_min_amt <= result['Amount'] <= weighted_max_amt
            result['Threshold_Range_Amount'] = f"₹{icici_amt_val.format_indian_currency(amt_stats.get('overall_min', 0))} - ₹{icici_amt_val.format_indian_currency(amt_stats.get('overall_max', 0))}"
            result['Weighted_Avg_Range_Amount'] = f"₹{icici_amt_val.format_indian_currency(weighted_min_amt)} - ₹{icici_amt_val.format_indian_currency(weighted_max_amt)}"
        else:
            result['Is_Valid_Amount'] = 'N/A'
            result['Threshold_Range_Amount'] = 'N/A'
            result['Weighted_Avg_Range_Amount'] = 'N/A'
    else:
        result['Is_Valid_Amount'] = 'N/A'
        result['Threshold_Range_Amount'] = 'N/A'
        result['Weighted_Avg_Range_Amount'] = 'N/A'
    
    # ===== RATE (using RATE model) =====
    if icici_sqft_rate in icici_rate_val.sqfeet_ranges:
        range_data = icici_rate_val.sqfeet_ranges[icici_sqft_rate]
        range_key = 'rate_ranges' if 'rate_ranges' in range_data else 'amount_ranges' if 'amount_ranges' in range_data else 'quantity_ranges'
        
        if range_key in range_data and icici_cleaned in range_data[range_key]:
            rate_stats = range_data[range_key][icici_cleaned]
            weighted_min_rate = rate_stats.get('weighted_avg_min', rate_stats.get('overall_mean', 0))
            weighted_max_rate = rate_stats.get('weighted_avg_max', rate_stats.get('overall_mean', 0))
            
            result['Is_Valid_Rate'] = weighted_min_rate <= result['Rate'] <= weighted_max_rate
            result['Threshold_Range_Rate'] = f"₹{icici_rate_val.format_indian_currency(rate_stats.get('overall_min', 0))} - ₹{icici_rate_val.format_indian_currency(rate_stats.get('overall_max', 0))}"
            result['Weighted_Avg_Range_Rate'] = f"₹{icici_rate_val.format_indian_currency(weighted_min_rate)} - ₹{icici_rate_val.format_indian_currency(weighted_max_rate)}"
        else:
            result['Is_Valid_Rate'] = 'N/A'
            result['Threshold_Range_Rate'] = 'N/A'
            result['Weighted_Avg_Range_Rate'] = 'N/A'
    else:
        result['Is_Valid_Rate'] = 'N/A'
        result['Threshold_Range_Rate'] = 'N/A'
        result['Weighted_Avg_Range_Rate'] = 'N/A'
    
    if result.get('Is_Valid_Qty') == True:
        result['Validation_Message_Qty'] = f"✅ Within range: {result.get('Weighted_Avg_Range_Qty', 'N/A')}"
    elif result.get('Is_Valid_Qty') == False:
        result['Validation_Message_Qty'] = f"❌ Outside range: {result.get('Weighted_Avg_Range_Qty', 'N/A')}"
    else:
        result['Validation_Message_Qty'] = "N/A - No validation data"
    
    if result.get('Is_Valid_Amount') == True:
        result['Validation_Message_Amount'] = f"✅ Within range: {result.get('Weighted_Avg_Range_Amount', 'N/A')}"
    elif result.get('Is_Valid_Amount') == False:
        result['Validation_Message_Amount'] = f"❌ Outside range: {result.get('Weighted_Avg_Range_Amount', 'N/A')}"
    else:
        result['Validation_Message_Amount'] = "N/A - No validation data"
    
    if result.get('Is_Valid_Rate') == True:
        result['Validation_Message_Rate'] = f"✅ Within range: {result.get('Weighted_Avg_Range_Rate', 'N/A')}"
    elif result.get('Is_Valid_Rate') == False:
        result['Validation_Message_Rate'] = f"❌ Outside range: {result.get('Weighted_Avg_Range_Rate', 'N/A')}"
    else:
        result['Validation_Message_Rate'] = "N/A - No validation data"
    
    return result
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if not VALIDATORS:
        return render_template('error.html', message="No validation models are available.")

    if request.method == 'POST':
        print("\n" + "="*60)
        print("🔵 UPLOAD REQUEST RECEIVED")
        print("="*60)
        
        # ✅ STEP 1: Check selected clients FIRST
        selected_clients = request.form.get('selected_clients', '')
        print(f"📋 Selected clients: {selected_clients}")

        # Get selected region (new)
        selected_region = request.form.get('selected_region', '')
        print(f"📍 Selected region: {selected_region}")
        
        # ✅ STEP 2: Detect comparison mode BEFORE any file checks
        is_comparison_mode = 'HDFC Bank' in selected_clients and 'ICICI Bank' in selected_clients
        print(f"🔄 Comparison mode: {is_comparison_mode}")
        
        # ✅ STEP 3: Handle comparison mode separately
        if is_comparison_mode:
            print("🔄 COMPARISON MODE DETECTED")
            
            # Check for dual files
            if 'hdfc_file' not in request.files or 'icici_file' not in request.files:
                print("❌ Both files required for comparison")
                flash('Both HDFC and ICICI files are required for comparison')
                return redirect(request.url)
            
            hdfc_file = request.files['hdfc_file']
            icici_file = request.files['icici_file']
            
            print(f"📁 HDFC file received: {hdfc_file.filename}")
            print(f"📁 ICICI file received: {icici_file.filename}")
            
            if hdfc_file.filename == '' or icici_file.filename == '':
                print("❌ File selection incomplete")
                flash('Please select both files')
                return redirect(request.url)
            
            if not (allowed_file(hdfc_file.filename) and allowed_file(icici_file.filename)):
                print("❌ Invalid file types")
                flash('Both files must be Excel format (.xlsx or .xls)')
                return redirect(request.url)
            
            # Save both files
            hdfc_filename = secure_filename(hdfc_file.filename)
            icici_filename = secure_filename(icici_file.filename)
            hdfc_filepath = os.path.join(app.config['UPLOAD_FOLDER'], hdfc_filename)
            icici_filepath = os.path.join(app.config['UPLOAD_FOLDER'], icici_filename)
            
            print(f"💾 Saving HDFC file: {hdfc_filename}")
            print(f"💾 Saving ICICI file: {icici_filename}")
            
            try:
                hdfc_file.save(hdfc_filepath)
                icici_file.save(icici_filepath)
                print("✅ Both files saved successfully")
            except Exception as e:
                print(f"❌ Error saving files: {str(e)}")
                flash(f'Error saving files: {str(e)}')
                return redirect(request.url)
            
            # ✅ NEW: Process QAR Analysis FIRST
            print("🔍 Starting QAR Analysis...")
            qar_data, qar_error = process_qar_analysis(hdfc_filepath, icici_filepath)
            
            if qar_error:
                print(f"❌ QAR Analysis error: {qar_error}")
                flash(qar_error)
                return redirect(request.url)
            
            print("✅ QAR Analysis completed successfully")
            
            # Also process grouped comparison (existing functionality)
            print("🔍 Starting grouped comparison processing...")
            comparison_data, error_msg = process_comparison(hdfc_filepath, icici_filepath)
            
            if error_msg:
                print(f"❌ Comparison error: {error_msg}")
                flash(error_msg)
                return redirect(request.url)
            
            print("✅ Grouped comparison completed successfully")
            print("="*60 + "\n")
            
            # ✅ NEW: Render QAR analysis template instead of comparison template
            return render_template('qar_analysis.html',
                                 qar_data=qar_data,
                                 comparison_data=comparison_data)
        
        # ✅ STEP 4: Single file upload (only runs if NOT comparison mode)
        print("📄 SINGLE FILE MODE")
        
        if 'file' not in request.files:
            print("❌ No file part in request")
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        print(f"📁 File received: {file.filename}")
        
        if file.filename == '':
            print("❌ No file selected")
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            print(f"✅ File type allowed: {file.filename}")
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"📂 Target filepath: {filepath}")
            
            # Remove existing file if it exists
            if os.path.exists(filepath):
                print(f"⚠️  File already exists, attempting to remove...")
                try:
                    os.remove(filepath)
                    print(f"✅ Old file removed successfully")
                except Exception as e:
                    print(f"❌ Error removing old file: {str(e)}")
                    flash(f'Error removing old file: {str(e)}')
                    return redirect(request.url)
            
            # Save new file
            print(f"💾 Saving file...")
            try:
                file.save(filepath)
                print(f"✅ File saved successfully to: {filepath}")
            except PermissionError as e:
                print(f"❌ Permission error: {str(e)}")
                flash('⚠️ File is currently open. Please close it and try again.')
                return redirect(request.url)
            except Exception as e:
                print(f"❌ Unexpected error saving file: {str(e)}")
                flash(f'Error saving file: {str(e)}')
                return redirect(request.url)
            
            # Get selected client
            selected_clients = request.form.get('selected_clients', 'HDFC Bank')
            print(f"🏦 Selected clients from form: {selected_clients}")
            
            client_name = selected_clients.split(',')[0] if selected_clients else 'HDFC Bank'
            validation_mode = request.form.get('validation_mode', 'quantity')
            session['last_uploaded_file'] = filepath
            session['last_client'] = client_name
            session['last_region'] = selected_region
            print(f"📊 Validation mode: {validation_mode}")

            is_regional = False

            # Build the model key based on client and mode
            # Build model key based on client and mode
            if client_name == 'HDFC Bank':
                if validation_mode == 'quantity':
                    model_key = 'hdfc_qty'
                elif validation_mode == 'amount':
                    model_key = 'hdfc_amount'
                elif validation_mode == 'rate':
                    model_key = 'hdfc_rate'
                elif validation_mode == 'grouped':
                    model_key = 'hdfc_grouped'
                else:
                    model_key = 'hdfc_qty'

            elif client_name == 'ICICI Bank':
                # ✅ NEW: Check if region is selected for ICICI
                is_regional = False  
                if selected_region and selected_region in ['east', 'west', 'north', 'south']:
                    is_regional = True
                    print(f"🗺️ REGIONAL MODE DETECTED: {selected_region.upper()}")

                    # ⚠️ PER USER REQUEST: Regional Quantity models are not trained yet.
                    # Force ALL regional requests to use 'rate' mode for now.
                    if validation_mode != 'rate':
                        print(f"⚠️ Mode '{validation_mode}' requested, but Regional Quantity model is not ready. Switching to 'rate'.")
                        validation_mode = 'rate'

                    # Always use the Rate model for regions
                    model_key = f'icici_rate_{selected_region}'
                        
                else:
                    # Use default ICICI models (no region selected)
                    if validation_mode == 'quantity':
                        model_key = 'icici_qty'
                    elif validation_mode == 'amount':
                        model_key = 'icici_amount'
                    elif validation_mode == 'rate':
                        model_key = 'icici_rate'
                    elif validation_mode == 'grouped':
                        model_key = 'icici_grouped'
                    else:
                        model_key = 'icici_qty'
            else:
                model_key = 'hdfc_qty'

            print(f"🏦 Using model key: {model_key}")

            # Handle grouped mode differently
            if validation_mode == 'grouped':
                if model_key not in GROUPED_VALIDATORS:
                    print(f"❌ Grouped validator not available")
                    flash(f'Grouped validation model is not available.')
                    return redirect(request.url)

                print(f"✅ Using grouped validator")
                VALIDATOR = GROUPED_VALIDATORS[model_key]
                results_data, error_msg = validate_grouped_file(VALIDATOR, filepath)
            else:
                if model_key not in VALIDATORS:
                    print(f"❌ Validator not available for model: {model_key}")
                    flash(f'Validation model for {client_name} - {validation_mode} is not available.')
                    return redirect(request.url)

                print(f"✅ Using {model_key} validator")
                VALIDATOR = VALIDATORS[model_key]
                selected_region = request.form.get('selected_region', '')
                # ✅ ADD THESE DEBUG LINES
                print(f"📂 Validating file: {os.path.basename(filepath)}")
                print(f"📊 Validation mode: {validation_mode}")
                print(f"📍 Selected region: {selected_region}")
                print(f"🎯 Model key: {model_key}")
                print(f"🌍 Is regional: {is_regional}")

                # Check what ranges are in the model
                if hasattr(VALIDATOR, 'sqfeet_ranges'):
                    print(f"🗂️  Model contains these ranges: {list(VALIDATOR.sqfeet_ranges.keys())}")

                results_data, error_msg = VALIDATOR.validate_test_file(filepath, mode=validation_mode, is_regional=is_regional)
            
            if error_msg:
                print(f"❌ Validation error: {error_msg}")
                flash(error_msg)
                return redirect(request.url)
            
            if results_data is None:
                print(f"❌ Validation returned None (no results)")
                flash("Validation failed - no results returned")
                return redirect(request.url)
            
            print(f"✅ Validation successful! Results count: {len(results_data)}")

            # Process results
            print(f"📊 Processing results...")
            results_df = pd.DataFrame(results_data)

            # ✅ FIX: DON'T extract thresholds from Validation_Message
            # The Threshold_Min/Max are already in results_data from validate_quantity()
            # Only extract for non-quantity modes that don't have these columns
            # Grouped mode doesn't have Validation_Message column
            if validation_mode not in ['quantity', 'grouped'] and 'Threshold_Min' not in results_df.columns:
                results_df['Threshold_Min'], results_df['Threshold_Max'] = zip(
                    *results_df['Validation_Message'].map(extract_thresholds)
                )

            # Save reports
            print(f"💾 Saving reports...")
            full_report_filename = f"validated_{filename}"
            full_report_filepath = os.path.join(app.config['OUTPUT_FOLDER'], full_report_filename)
            styled_df = results_df.style.apply(highlight_invalid, axis=1)
            styled_df.to_excel(full_report_filepath, engine='xlsxwriter', index=False)
            print(f"✅ Full report saved: {full_report_filename}")
            
            invalid_df = results_df[results_df['Is_Valid'] == False].copy()
            invalid_report_filename = f"invalid_rows_{filename}"
            if not invalid_df.empty:
                invalid_report_filepath = os.path.join(app.config['OUTPUT_FOLDER'], invalid_report_filename)
                invalid_df.to_excel(invalid_report_filepath, index=False)
                print(f"✅ Invalid report saved: {invalid_report_filename}")

            # Prepare display
            print(f"🖥️  Preparing web display...")
            display_df = results_df.copy()

            # Format display based on validation_mode
            if validation_mode == 'grouped':
                # Format CWI Amount
                display_df['As_Per_CWI_Amount'] = display_df['As_Per_CWI_Amount'].apply(
                    lambda x: VALIDATOR.format_indian_currency(x)
                )
                
                # Format Rate per sqfeet
                display_df['Rate_per_sqfeet'] = display_df['Rate_per_sqfeet'].apply(
                    lambda x: VALIDATOR.format_indian_currency(x)
                )
                
                # ✅ NEW: Format CWI Weighted Averages
                if 'CWI_Weighted_Min' in display_df.columns:
                    def format_cwi_weighted(row, col_name):
                        val = row.get(col_name)
                        if val == 'NA' or pd.isna(val):
                            return 'NA'
                        return VALIDATOR.format_indian_currency(val)
                    
                    display_df['CWI_Weighted_Min'] = display_df.apply(
                        lambda row: format_cwi_weighted(row, 'CWI_Weighted_Min'), axis=1
                    )
                    display_df['CWI_Weighted_Max'] = display_df.apply(
                        lambda row: format_cwi_weighted(row, 'CWI_Weighted_Max'), axis=1
                    )
                
                # ✅ NEW: Format Rate Weighted Averages
                if 'Rate_Weighted_Min' in display_df.columns:
                    def format_rate_weighted(row, col_name):
                        val = row.get(col_name)
                        if val == 'NA' or pd.isna(val):
                            return 'NA'
                        return VALIDATOR.format_indian_currency(val)
                    
                    display_df['Rate_Weighted_Min'] = display_df.apply(
                        lambda row: format_rate_weighted(row, 'Rate_Weighted_Min'), axis=1
                    )
                    display_df['Rate_Weighted_Max'] = display_df.apply(
                        lambda row: format_rate_weighted(row, 'Rate_Weighted_Max'), axis=1
                    )

            elif validation_mode == 'rate':
                # Store raw numeric values BEFORE formatting (for analysis button)
                display_df['Rate_Raw'] = display_df['Rate'].copy()
                display_df['Amount_Raw'] = display_df['Amount'].copy()

                # 1. Format Rate Column with Indian Currency Style
                def format_rate_for_display(row):
                    rate = row['Rate']
                    return VALIDATOR.format_indian_currency(rate)

                display_df['Rate'] = display_df.apply(format_rate_for_display, axis=1)

                # 2. Format Amount Column with Indian Currency Style
                def format_amount_for_display(row):
                    amount = row['Amount']
                    return VALIDATOR.format_indian_currency(amount)

                display_df['Amount'] = display_df.apply(format_amount_for_display, axis=1)

                # 3. Round Quantity if unit is number-based
                def format_qty(row):
                    unit = row['Unit']
                    qty = row['Quantity']
                    if VALIDATOR.is_number_based_unit(unit):
                        return int(round(qty))
                    return qty

                display_df['Quantity'] = display_df.apply(format_qty, axis=1)

                # 4. Format Weighted Averages for rate mode
                if 'Weighted_Avg_Min' in display_df.columns:
                    def format_weighted_avg(row, col_name):
                        val = row.get(col_name)
                        if pd.isna(val) or val is None or val == 'NA':
                            return 'NA'
                        # Always format as currency for rate mode
                        return VALIDATOR.format_indian_currency(val)

                    display_df['Weighted_Avg_Min'] = display_df.apply(
                        lambda row: format_weighted_avg(row, 'Weighted_Avg_Min'), axis=1
                    )
                    display_df['Weighted_Avg_Max'] = display_df.apply(
                        lambda row: format_weighted_avg(row, 'Weighted_Avg_Max'), axis=1
                    )

                # 5. Format Thresholds as clickable elements with file information
                if 'Threshold_Min' in display_df.columns and 'Threshold_Max' in display_df.columns:
                    def format_threshold_clickable(row, col_name, files_col_name):
                        val = row.get(col_name)
                        if val == 'NA' or pd.isna(val):
                            return 'NA'

                        # Format as currency
                        display_val = VALIDATOR.format_indian_currency(val)

                        # Get file information
                        files_data = row.get(files_col_name, [])
                        if not files_data or len(files_data) == 0:
                            return str(display_val)

                        # Create tooltip text with file information
                        tooltip_lines = []
                        for i, file_info in enumerate(files_data[:5]):
                            file_name = file_info.get('file', 'Unknown')
                            particular = file_info.get('original_particular', 'N/A')
                            tooltip_lines.append(f"{i+1}. {file_name}: {particular}")

                        if len(files_data) > 5:
                            tooltip_lines.append(f"... and {len(files_data) - 5} more files")

                        tooltip_text = "\\n".join(tooltip_lines)
                        tooltip_text = html.escape(tooltip_text)

                        return f'<span class="threshold-clickable" title="{tooltip_text}" style="cursor: pointer; color: #007bff; text-decoration: underline;">{display_val}</span>'

                    display_df['Threshold_Min'] = display_df.apply(
                        lambda row: format_threshold_clickable(row, 'Threshold_Min', 'Min_Threshold_Files'), axis=1
                    )
                    display_df['Threshold_Max'] = display_df.apply(
                        lambda row: format_threshold_clickable(row, 'Threshold_Max', 'Max_Threshold_Files'), axis=1
                    )

            elif validation_mode == 'amount':
                # Store raw numeric values BEFORE formatting
                display_df['Amount_Raw'] = display_df['Amount'].copy()

                # Format the Main Amount Column
                def format_amount_for_display(row):
                    unit = row['Unit']
                    amount = row['Amount']
                    if VALIDATOR.is_number_based_unit(unit):
                        val = int(round(amount))
                    else:
                        val = amount
                    return VALIDATOR.format_indian_currency(val)

                display_df['Amount'] = display_df.apply(format_amount_for_display, axis=1)

                # Format Weighted Averages for amount mode
                if 'Weighted_Avg_Min' in display_df.columns:
                    def format_weighted_avg(row, col_name):
                        val = row.get(col_name)
                        if pd.isna(val) or val is None or val == 'NA':
                            return 'NA'
                        # Always format as currency for amount mode
                        return VALIDATOR.format_indian_currency(val)

                    display_df['Weighted_Avg_Min'] = display_df.apply(
                        lambda row: format_weighted_avg(row, 'Weighted_Avg_Min'), axis=1
                    )
                    display_df['Weighted_Avg_Max'] = display_df.apply(
                        lambda row: format_weighted_avg(row, 'Weighted_Avg_Max'), axis=1
                    )

                # Format Thresholds as clickable elements with file information
                if 'Threshold_Min' in display_df.columns and 'Threshold_Max' in display_df.columns:
                    def format_threshold_clickable(row, col_name, files_col_name):
                        val = row.get(col_name)
                        if val == 'NA' or pd.isna(val):
                            return 'NA'

                        # Format as currency
                        display_val = VALIDATOR.format_indian_currency(val)

                        # Get file information
                        files_data = row.get(files_col_name, [])
                        if not files_data or len(files_data) == 0:
                            return str(display_val)

                        # Create tooltip text with file information
                        tooltip_lines = []
                        for i, file_info in enumerate(files_data[:5]):
                            file_name = file_info.get('file', 'Unknown')
                            particular = file_info.get('original_particular', 'N/A')
                            tooltip_lines.append(f"{i+1}. {file_name}: {particular}")

                        if len(files_data) > 5:
                            tooltip_lines.append(f"... and {len(files_data) - 5} more files")

                        tooltip_text = "\\n".join(tooltip_lines)
                        tooltip_text = html.escape(tooltip_text)

                        return f'<span class="threshold-clickable" title="{tooltip_text}" style="cursor: pointer; color: #007bff; text-decoration: underline;">{display_val}</span>'

                    display_df['Threshold_Min'] = display_df.apply(
                        lambda row: format_threshold_clickable(row, 'Threshold_Min', 'Min_Threshold_Files'), axis=1
                    )
                    display_df['Threshold_Max'] = display_df.apply(
                        lambda row: format_threshold_clickable(row, 'Threshold_Max', 'Max_Threshold_Files'), axis=1
                    )

            else:  # quantity mode
                def format_quantity_for_display(row):
                    unit = row['Unit']
                    quantity = row['Quantity']
                    if VALIDATOR.is_number_based_unit(unit):
                        return int(round(quantity))
                    return quantity

                display_df['Quantity'] = display_df.apply(format_quantity_for_display, axis=1)

                # Format Weighted Averages for quantity mode
                if 'Weighted_Avg_Min' in display_df.columns:
                    def format_weighted_avg(row, col_name):
                        val = row.get(col_name)
                        if pd.isna(val) or val is None or val == 'NA':
                            return 'NA'
                        unit = row.get('Unit', '')
                        if VALIDATOR.is_number_based_unit(unit):
                            return int(round(val))
                        return f"{val:.2f}"

                    display_df['Weighted_Avg_Min'] = display_df.apply(
                        lambda row: format_weighted_avg(row, 'Weighted_Avg_Min'), axis=1
                    )
                    display_df['Weighted_Avg_Max'] = display_df.apply(
                        lambda row: format_weighted_avg(row, 'Weighted_Avg_Max'), axis=1
                    )

                # Format Thresholds as clickable elements with file information
                if 'Threshold_Min' in display_df.columns and 'Threshold_Max' in display_df.columns:
                    def format_threshold_clickable(row, col_name, files_col_name):
                        val = row.get(col_name)
                        if val == 'NA' or pd.isna(val):
                            return 'NA'

                        unit = row.get('Unit', '')
                        if VALIDATOR.is_number_based_unit(unit):
                            display_val = int(round(float(val)))
                        else:
                            display_val = f"{float(val):.2f}"

                        files_data = row.get(files_col_name, [])
                        if not files_data or len(files_data) == 0:
                            return str(display_val)

                        tooltip_lines = []
                        for i, file_info in enumerate(files_data[:5]):
                            file_name = file_info.get('file', 'Unknown')
                            particular = file_info.get('original_particular', 'N/A')
                            tooltip_lines.append(f"{i+1}. {file_name}: {particular}")

                        if len(files_data) > 5:
                            tooltip_lines.append(f"... and {len(files_data) - 5} more files")

                        tooltip_text = "\\n".join(tooltip_lines)
                        tooltip_text = html.escape(tooltip_text)

                        return f'<span class="threshold-clickable" title="{tooltip_text}" style="cursor: pointer; color: #007bff; text-decoration: underline;">{display_val}</span>'

                    display_df['Threshold_Min'] = display_df.apply(
                        lambda row: format_threshold_clickable(row, 'Threshold_Min', 'Min_Threshold_Files'), axis=1
                    )
                    display_df['Threshold_Max'] = display_df.apply(
                        lambda row: format_threshold_clickable(row, 'Threshold_Max', 'Max_Threshold_Files'), axis=1
                    )

            # Add hover tooltip for Original_Particular column
            if 'Original_Particular' in display_df.columns:
                display_df['Original_Particular'] = display_df['Original_Particular'].apply(
                    lambda x: f'<span title="{html.escape(str(x))}">{html.escape(str(x))}</span>'
                )    

            # Analysis button - only for non-grouped modes
            if validation_mode != 'grouped':
                def create_analysis_button(row):
                    if not row['Is_Valid'] and row['Threshold_Min'] != 'NA':
                        safe_particular = html.escape(row["Original_Particular"])
                        if validation_mode == 'rate':
                            value = row.get('Rate_Raw', 0)
                        elif validation_mode == 'amount':
                            value = row.get('Amount_Raw', 0)
                        else:
                            value = row.get('Quantity', 0)

                        # ✅ NEW: Add region parameter to button
                        return (f'<button class="btn btn-sm btn-info btn-analysis" '
                                f'data-particular="{safe_particular}" '
                                f'data-quantity="{value}" '
                                f'data-range="{row["Sq_Feet_Range"]}" '
                                f'data-client="{client_name}" '
                                f'data-mode="{validation_mode}" '
                                f'data-region="{selected_region}">'
                                f'View Analysis</button>')
                    return ''

                display_df['Analysis'] = display_df.apply(create_analysis_button, axis=1)

            display_df['Is_Valid'] = display_df['Is_Valid'].apply(lambda x: '✅ Valid' if x else '❌ Invalid')
            
            # Dynamic column order based on mode
            # Dynamic column order based on mode
            if validation_mode == 'grouped':
                column_order = ['Sr.No', 'Particulars', 'As_Per_CWI_Amount', 'Rate_per_sqfeet',
                               'Sq_Feet_Range', 'Is_Valid', 
                               'CWI_Validation', 'Rate_Validation',
                               'CWI_Weighted_Min', 'CWI_Weighted_Max',
                               'Rate_Weighted_Min', 'Rate_Weighted_Max']
            elif validation_mode == 'amount':
                column_order = ['Sr.No', 'Original_Particular', 'Amount', 'Unit', 
                               'Sq_Feet_Range', 'Is_Valid', 'Validation_Message', 
                               'Matched_Particular', 'Similarity_Score', 
                               'Threshold_Min', 'Threshold_Max',
                               'Weighted_Avg_Min', 'Weighted_Avg_Max',
                               'Analysis']
            elif validation_mode == 'rate':
                if is_regional:
                    column_order = ['Sr.No', 'Original_Particular', 'Rate', 'Unit', 'Quantity', 'Amount',
                                   'Sq_Feet_Range', 'Is_Valid', 
                                   'Validation_Message_East', 'Validation_Message_West', 
                                   'Validation_Message_North', 'Validation_Message_South',
                                   'Matched_Particular', 'Similarity_Score', 
                                   'Threshold_Min', 'Threshold_Max',
                                   'Weighted_Avg_Min', 'Weighted_Avg_Max',
                                   'Analysis']
                else:
                    column_order = ['Sr.No', 'Original_Particular', 'Rate', 'Unit', 'Quantity', 'Amount',
                                   'Sq_Feet_Range', 'Is_Valid', 'Validation_Message', 
                                   'Matched_Particular', 'Similarity_Score', 
                                   'Threshold_Min', 'Threshold_Max',
                                   'Weighted_Avg_Min', 'Weighted_Avg_Max',
                                   'Analysis']
            else:  # quantity mode - ✅ THIS IS NOW CORRECT
                column_order = ['Sr.No', 'Original_Particular', 'Quantity', 'Unit', 
                                    'Sq_Feet_Range', 'Is_Valid', 'Validation_Message', 
                                    'Matched_Particular', 'Similarity_Score', 
                                    'Threshold_Min', 'Threshold_Max',
                                    'Weighted_Avg_Min', 'Weighted_Avg_Max',  
                                    'Analysis']
            display_df = display_df[column_order]
            
            # Generate HTML table
            results_html = display_df.to_html(classes='table table-striped table-hover', justify='left', index=False, escape=False)

            # ✅ FIX: Highlight the SELECTED REGION header in Green
            if is_regional and selected_region:
                # Construct the specific column name found in the DataFrame
                region_col_name = f'Validation_Message_{selected_region.capitalize()}'

                # Inject the CSS class into the HTML for that specific header
                # We replace <th>ColumnName</th> with <th class="selected-region-header">ColumnName ✓</th>
                results_html = results_html.replace(
                    f'<th>{region_col_name}</th>',
                    f'<th class="selected-region-header">{region_col_name} <i class="fas fa-check-circle"></i></th>'
                )

            # ✅ FIX: Add CSS class to selected region column if regional mode
            if is_regional and selected_region:
                region_col_name = f'Validation_Message_{selected_region.capitalize()}'

                # Add green header class to selected region
                results_html = results_html.replace(
                    f'<th>{region_col_name}</th>',
                    f'<th class="selected-region-header">{region_col_name} ✓</th>'
                )
    
    
            
            total = len(results_df)
            valid = results_df['Is_Valid'].sum()
            invalid = total - valid
            accuracy = (valid / total * 100) if total > 0 else 0
            
            print(f"📈 Statistics: Total={total}, Valid={valid}, Invalid={invalid}, Accuracy={accuracy:.2f}%")
            print(f"✅ Rendering results page...")
            print("="*60 + "\n")

            return render_template('results.html', 
                    table_html=results_html, 
                    full_report_filename=full_report_filename,
                    invalid_report_filename=invalid_report_filename,
                    total=total, valid=valid, invalid=invalid, 
                    accuracy=f"{accuracy:.2f}%",
                    client=client_name,
                    validation_mode=validation_mode,
                    region=selected_region)  # ✅ THIS LINE MUST BE HERE
        else:
            print(f"❌ Invalid file type: {file.filename}")
            flash('Invalid file type. Please upload an .xlsx or .xls file.')
            return redirect(request.url)

    return render_template('index.html')
@app.route('/outputs/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
    
@app.route('/error')
def error_page():
    return render_template('error.html', message="An unknown error occurred.")

# --- FINAL /graph ROUTE with ALL fixes ---
@app.route('/graph')
def get_graph():
    try:
        # --- 1. Get data from request ---
        chart_type = request.args.get('type')
        sq_range = request.args.get('range')
        particular = request.args.get('particular')
        user_value = float(request.args.get('value', 0))
        client = request.args.get('client', 'HDFC Bank')
        validation_mode = request.args.get('mode', 'quantity')
        
        # ✅ NEW: Get region parameter for regional models
        selected_region = request.args.get('region', '')

        print(f"🔍 Graph request: {chart_type}, range={sq_range}, mode={validation_mode}, region={selected_region}")

        # Build model key
        if client == 'HDFC Bank':
            if validation_mode == 'quantity':
                model_key = 'hdfc_qty'
            elif validation_mode == 'amount':
                model_key = 'hdfc_amount'
            elif validation_mode == 'rate':
                model_key = 'hdfc_rate'
            else:
                model_key = 'hdfc_qty'
        elif client == 'ICICI Bank':
            # ✅ NEW: Check for regional mode
            if selected_region and selected_region in ['east', 'west', 'north', 'south']:
                # Use regional model
                if validation_mode == 'rate':
                    model_key = f'icici_rate_{selected_region}'
                elif validation_mode == 'quantity':
                    model_key = f'icici_qty_{selected_region}'
                else:
                    model_key = f'icici_rate_{selected_region}'  # Default to rate for regional
                print(f"🗺️ Using REGIONAL model: {model_key}")
            else:
                # Use standard models
                if validation_mode == 'quantity':
                    model_key = 'icici_qty'
                elif validation_mode == 'amount':
                    model_key = 'icici_amount'
                elif validation_mode == 'rate':
                    model_key = 'icici_rate'
                else:
                    model_key = 'icici_qty'
        else:
            model_key = 'hdfc_qty'

        print(f"🔑 Using model: {model_key}")

        if model_key not in VALIDATORS:
            return jsonify({'error': f'Validation model not available for {model_key}'})

        validator = VALIDATORS[model_key]

        # FIXED: Determine which key to use
        if 'rate_ranges' in validator.sqfeet_ranges.get(sq_range, {}):
            range_key = 'rate_ranges'
        elif 'amount_ranges' in validator.sqfeet_ranges.get(sq_range, {}):
            range_key = 'amount_ranges'
        else:
            range_key = 'quantity_ranges'

        cleaned_particular = validator.clean_text(particular)
        matched_particular = None
        
        # FIXED: Use range_key instead of hardcoded 'quantity_ranges'
        if cleaned_particular in validator.sqfeet_ranges[sq_range][range_key]:
            matched_particular = cleaned_particular
        else:
            sim_p, _ = validator.find_similar_particular(particular, sq_range)
            if sim_p: 
                matched_particular = sim_p

        if not matched_particular:
            return jsonify({'error': 'Could not find statistics for this item.'})

        # FIXED: Use range_key instead of hardcoded 'quantity_ranges'
        stats = validator.sqfeet_ranges[sq_range][range_key][matched_particular]
        mean = stats.get('overall_mean', 0)
        std = stats.get('overall_std', 0)

        # ✅ FIX: Use THRESHOLD values (absolute min/max) for quantity mode
        # For quantity mode, use threshold values (overall_min/max)
        # For amount/rate modes, use validation_range which is correct
        # REPLACE WITH:
        if validation_mode == 'quantity':
            lower_bound = stats.get('weighted_avg_min', stats.get('overall_mean', 0))
            upper_bound = stats.get('weighted_avg_max', stats.get('overall_mean', 0))  
        else:
            # ✅ USE WEIGHTED AVERAGES FOR AMOUNT AND RATE
            lower_bound = stats.get('weighted_avg_min', stats.get('overall_mean', 0))
            upper_bound = stats.get('weighted_avg_max', stats.get('overall_mean', 0))
        # --- 3. Generate the correct plot based on chart_type ---
        fig, ax = plt.subplots(figsize=(8, 4))
        explanation = ""

        if chart_type == 'bell_curve':
            if std == 0: std = mean * 0.1 if mean > 0 else 1
            x = np.linspace(mean - 4*std, mean + 4*std, 1000)
            y = norm.pdf(x, mean, std)
            ax.plot(x, y, label='Normal Distribution')
            ax.axvline(lower_bound, color='green', linestyle='--', label=f'Valid Min: {lower_bound:.2f}')
            ax.axvline(upper_bound, color='green', linestyle='--')
            ax.axvline(mean, color='orange', linestyle='-', label=f'Average: {mean:.2f}')
            ax.plot(user_value, 0, 'rX', markersize=12, label=f'Your Value: {user_value:.2f}')
            ax.fill_between(x, y, where=((x >= lower_bound) & (x <= upper_bound)), color='green', alpha=0.2)
            ax.set_title('Bell Curve Analysis')
            ax.set_xlabel('Quantity')
            ax.set_ylabel('Likelihood')
            # ✅ FIX: Update explanation to show THRESHOLD range for quantity mode
            if validation_mode == 'quantity':
                # ✅ UNIFIED EXPLANATION FOR ALL MODES
                explanation = (f"This bell curve shows the expected distribution. The average is <b>{mean:.2f}</b>, and the weighted average range is from <b>{lower_bound:.2f}</b> to <b>{upper_bound:.2f}</b>. "
                               f"Your value of <b style='color:red;'>{user_value:.2f}</b> is outside this range.")
            elif validation_mode in ['amount', 'rate']:
                # ✅ NEW: Different explanation for weighted averages
                explanation = (f"This bell curve shows the expected distribution. The average is <b>{mean:.2f}</b>, and the weighted average range (validated range from historical data) is from <b>{lower_bound:.2f}</b> to <b>{upper_bound:.2f}</b>. "
                               f"Your value of <b style='color:red;'>{user_value:.2f}</b> is outside this range.")
            else:
                explanation = (f"This bell curve shows the expected distribution. The average is <b>{mean:.2f}</b>, and the valid range is from <b>{lower_bound:.2f}</b> to <b>{upper_bound:.2f}</b>. "
                               f"Your value of <b style='color:red;'>{user_value:.2f}</b> is outside this range.")

        elif chart_type == 'bar_plot':
            ax.bar(['Min', 'Max'], [lower_bound, upper_bound], color='green', alpha=0.6, label='Acceptable Range')
            ax.bar(['Your Value'], [user_value], color='red', label='Your Value')
            ax.set_title('Bar Plot Comparison')
            ax.set_ylabel('Quantity')
            if validation_mode == 'quantity':
                # ✅ UNIFIED EXPLANATION FOR ALL MODES
                explanation = (f"This bar plot compares your value against the weighted average min/max. "
                               f"Your value is <b style='color:red;'>{user_value:.2f}</b>, while the weighted average range is <b>{lower_bound:.2f}</b> - <b>{upper_bound:.2f}</b>.")
            elif validation_mode in ['amount', 'rate']:
                # ✅ UNIFIED EXPLANATION FOR ALL MODES
                explanation = (f"This bar plot compares your value against the weighted average min/max. "
                               f"Your value is <b style='color:red;'>{user_value:.2f}</b>, while the weighted average range is <b>{lower_bound:.2f}</b> - <b>{upper_bound:.2f}</b>.")
            else:
                # ✅ UNIFIED EXPLANATION FOR ALL MODES
                explanation = (f"This bar plot compares your value against the weighted average min/max. "
                               f"Your value is <b style='color:red;'>{user_value:.2f}</b>, while the weighted average range is <b>{lower_bound:.2f}</b> - <b>{upper_bound:.2f}</b>.")

        elif chart_type == 'box_plot':
            fig, ax = plt.subplots(figsize=(9, 5)) 
            if std == 0: std = mean * 0.1 if mean > 0 else 1
            np.random.seed(0) 
            sample_data = np.random.normal(loc=mean, scale=std, size=100)
            # Add the user's value to the data to see it as an outlier if it is one
            data_with_user_value = np.append(sample_data, user_value)
            
            boxplot = ax.boxplot(data_with_user_value, vert=False, widths=0.4, patch_artist=True,
                                 boxprops=dict(facecolor='lightblue'),
                                 medianprops=dict(color='red', linewidth=2))

            ax.plot(user_value, 1, 'rX', markersize=12, label=f'Your Value: {user_value:.2f}')
            ax.set_title('Box Plot Analysis (Historical Data Distribution)')
            ax.set_xlabel('Quantity')
            ax.grid(True, axis='x', linestyle='--', alpha=0.7) 
            ax.set_yticklabels([]) 
            ax.set_ylim(0.5, 1.5)
            explanation = (f"This Box Plot shows the statistical spread of historical data. The 'box' represents the middle 50% of the data. "
                           f"The red line inside the box is the median. The whiskers show the expected range, and circles are outliers. "
                           f"Your value of <b style='color:red;'>{user_value:.2f}</b> (the red 'X') is compared against this distribution.")

        else: # Default to threshold_line
            fig, ax = plt.subplots(figsize=(9, 3)) 
            ax.hlines(y=1, xmin=lower_bound, xmax=upper_bound, linewidth=15, color='green', alpha=0.5)
            
            is_valid = lower_bound <= user_value <= upper_bound
            marker_color = 'blue' if is_valid else 'red'
            ax.plot(user_value, 1, 'X', markersize=15, color=marker_color, label=f'Your Value: {user_value:.2f}')

            ax.text(lower_bound, 0.97, f'Min\n{lower_bound:.2f}', ha='center', va='top')
            ax.text(upper_bound, 0.97, f'Max\n{upper_bound:.2f}', ha='center', va='top')
            ax.text(user_value, 1.03, f'Your Value\n{user_value:.2f}', ha='center', va='bottom', color=marker_color, weight='bold')

            padding = (upper_bound - lower_bound) * 0.5 if (upper_bound > lower_bound) else upper_bound or 1
            ax.set_xlim(min(lower_bound, user_value) - padding, max(upper_bound, user_value) + padding)
            ax.set_ylim(0.9, 1.1) 
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            ax.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_title('Threshold Line Chart')
            plt.tight_layout()
            if validation_mode == 'quantity':
                # ✅ UNIFIED EXPLANATION FOR ALL MODES
                explanation = (f"The green line shows the weighted average range from <b>{lower_bound:.2f}</b> to <b>{upper_bound:.2f}</b>. "
                               f"Your value of <b style='color:{marker_color};'>{user_value:.2f}</b> is shown by the 'X'.")
            elif validation_mode in ['amount', 'rate']:
                # ✅ UNIFIED EXPLANATION FOR ALL MODES
                explanation = (f"The green line shows the weighted average range from <b>{lower_bound:.2f}</b> to <b>{upper_bound:.2f}</b>. "
                               f"Your value of <b style='color:{marker_color};'>{user_value:.2f}</b> is shown by the 'X'.")
            else:
                # ✅ UNIFIED EXPLANATION FOR ALL MODES
                explanation = (f"The green line shows the weighted average range from <b>{lower_bound:.2f}</b> to <b>{upper_bound:.2f}</b>. "
                               f"Your value of <b style='color:{marker_color};'>{user_value:.2f}</b> is shown by the 'X'.")
        
        if chart_type != 'threshold_line':
            ax.legend()
        
        # --- 4. Save plot to buffer and encode ---
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return jsonify({'image': image_base64, 'explanation': explanation})

    except Exception as e:
        print(f"Error in graph generation: {e}")
        return jsonify({'error': 'An error occurred while generating the chart.'})

@app.route('/switch_mode/<mode>')
def switch_mode(mode):
    """Switch validation mode and re-process the last uploaded file"""
    try:
        print("\n" + "="*60)
        print(f"🔄 MODE SWITCH REQUEST: {mode}")
        print("="*60)
        
        # Get stored file info from session
        filepath = session.get('last_uploaded_file')
        client_name = session.get('last_client', 'HDFC Bank')
        selected_region = session.get('last_region', '')

        is_regional = False
        
        print(f"📂 Session filepath: {filepath}")
        print(f"🏦 Session client: {client_name}")
        
        if not filepath:
            print("❌ No filepath in session!")
            flash('No file found in session. Please upload a new file.')
            return redirect(url_for('upload_file'))
            
        if not os.path.exists(filepath):
            print(f"❌ File does not exist at: {filepath}")
            flash('Uploaded file no longer exists. Please upload a new file.')
            return redirect(url_for('upload_file'))
        
        print(f"✅ File exists: {filepath}")
        
        # Build model key based on client and mode
        # Build model key based on client and mode
        # Build model key based on client and mode
        if client_name == 'HDFC Bank':
            if mode == 'quantity':  # ✅ CORRECT
                model_key = 'hdfc_qty'
            elif mode == 'amount':  # ✅ CORRECT
                model_key = 'hdfc_amount'
            elif mode == 'rate':
                model_key = 'hdfc_rate'
            elif mode == 'grouped':
                model_key = 'hdfc_grouped'
            else:
                model_key = 'hdfc_qty'

        elif client_name == 'ICICI Bank':
            # ✅ Check if region is selected for ICICI
            is_regional = False
            if selected_region and selected_region in ['east', 'west', 'north', 'south']:
                is_regional = True
                
                # ⚠️ PER USER REQUEST: Regional Quantity models are not trained yet.
                # Force ALL regional requests to use 'rate' mode for now.
                if mode != 'rate':
                    print(f"⚠️ Mode '{mode}' requested, but Regional Quantity model is not ready. Switching to 'rate'.")
                    mode = 'rate'  # Override the mode variable

                # Always use the Rate model for regions
                model_key = f'icici_rate_{selected_region}'

            else:
                # Use default ICICI models (no region selected)
                if mode == 'quantity':
                    model_key = 'icici_qty'
                elif mode == 'amount':
                    model_key = 'icici_amount'
                elif mode == 'rate':
                    model_key = 'icici_rate'
                elif mode == 'grouped':
                    model_key = 'icici_grouped'
                else:
                    model_key = 'icici_qty'
        else:
            model_key = 'hdfc_qty'
        
        print(f"🔑 Using model key: {model_key}")
        
        # Handle grouped mode
        if mode == 'grouped':
            if model_key not in GROUPED_VALIDATORS:
                print(f"❌ Grouped model not loaded")
                flash(f'Grouped validation model is not available.')
                return redirect(url_for('upload_file'))
            
            print(f"✅ Grouped validator loaded")
            VALIDATOR = GROUPED_VALIDATORS[model_key]
            
            print(f"🔍 Starting grouped validation...")
            results_data, error_msg = validate_grouped_file(VALIDATOR, filepath)
        else:
            if model_key not in VALIDATORS:
                print(f"❌ Model not loaded: {model_key}")
                flash(f'Validation model for {client_name} - {mode} is not available.')
                return redirect(url_for('upload_file'))
            
            print(f"✅ Validator loaded: {model_key}")
            VALIDATOR = VALIDATORS[model_key]
            
            # Validate file with mode parameter
            print(f"🔍 Starting validation with {mode} mode...")
            results_data, error_msg = VALIDATOR.validate_test_file(filepath, mode=mode, is_regional=is_regional)
        
        if error_msg:
            print(f"❌ Validation error: {error_msg}")
            flash(error_msg)
            return redirect(url_for('upload_file'))
        
        if results_data is None:
            print("❌ No results returned")
            flash("Validation failed - no results returned")
            return redirect(url_for('upload_file'))
        
        print(f"✅ Validation successful! Results: {len(results_data)} rows")
        
        # Process results
        results_df = pd.DataFrame(results_data)

        # ✅ FIX: DON'T extract thresholds for quantity, amount, or grouped modes
        # The Threshold_Min/Max are already in results_data for these modes
        if mode not in ['quantity', 'amount', 'grouped'] and 'Threshold_Min' not in results_df.columns:
            results_df['Threshold_Min'], results_df['Threshold_Max'] = zip(
                *results_df['Validation_Message'].map(extract_thresholds)
            )
        
        # Save reports
        filename = os.path.basename(filepath)
        full_report_filename = f"validated_{mode}_{filename}"
        full_report_filepath = os.path.join(app.config['OUTPUT_FOLDER'], full_report_filename)
        
        if mode != 'grouped':
            styled_df = results_df.style.apply(highlight_invalid, axis=1)
            styled_df.to_excel(full_report_filepath, engine='xlsxwriter', index=False)
        else:
            results_df.to_excel(full_report_filepath, index=False)
        
        print(f"✅ Report saved: {full_report_filename}")
        
        invalid_df = results_df[results_df['Is_Valid'] == False].copy()
        invalid_report_filename = f"invalid_{mode}_{filename}"
        if not invalid_df.empty:
            invalid_report_filepath = os.path.join(app.config['OUTPUT_FOLDER'], invalid_report_filename)
            invalid_df.to_excel(invalid_report_filepath, index=False)
            print(f"✅ Invalid report saved: {invalid_report_filename}")
        
        # Prepare display
        display_df = results_df.copy()
        
        if mode == 'grouped':
            # Apply Indian Currency Formatting
            display_df['As_Per_CWI_Amount'] = display_df['As_Per_CWI_Amount'].apply(
                lambda x: VALIDATOR.format_indian_currency(x)
            )
            display_df['Rate_per_sqfeet'] = display_df['Rate_per_sqfeet'].apply(
                lambda x: VALIDATOR.format_indian_currency(x)
            )

            # ✅ NEW: Format CWI Weighted Averages
            if 'CWI_Weighted_Min' in display_df.columns:
                def format_cwi_weighted(row, col_name):
                    val = row.get(col_name)
                    if val == 'NA' or pd.isna(val):
                        return 'NA'
                    return VALIDATOR.format_indian_currency(val)

                display_df['CWI_Weighted_Min'] = display_df.apply(
                    lambda row: format_cwi_weighted(row, 'CWI_Weighted_Min'), axis=1
                )
                display_df['CWI_Weighted_Max'] = display_df.apply(
                    lambda row: format_cwi_weighted(row, 'CWI_Weighted_Max'), axis=1
                )

            # ✅ NEW: Format Rate Weighted Averages
            if 'Rate_Weighted_Min' in display_df.columns:
                def format_rate_weighted(row, col_name):
                    val = row.get(col_name)
                    if val == 'NA' or pd.isna(val):
                        return 'NA'
                    return VALIDATOR.format_indian_currency(val)

                display_df['Rate_Weighted_Min'] = display_df.apply(
                    lambda row: format_rate_weighted(row, 'Rate_Weighted_Min'), axis=1
                )
                display_df['Rate_Weighted_Max'] = display_df.apply(
                    lambda row: format_rate_weighted(row, 'Rate_Weighted_Max'), axis=1
                )
            
        elif mode == 'rate':
            # Store raw numeric values BEFORE formatting (for analysis button)
            display_df['Rate_Raw'] = display_df['Rate'].copy()
            display_df['Amount_Raw'] = display_df['Amount'].copy()
        
            # 1. Format Rate Column with Indian Currency Style
            def format_rate_for_display(row):
                rate = row['Rate']
                return VALIDATOR.format_indian_currency(rate)
        
            display_df['Rate'] = display_df.apply(format_rate_for_display, axis=1)
        
            # 2. Format Amount Column with Indian Currency Style
            def format_amount_for_display(row):
                amount = row['Amount']
                return VALIDATOR.format_indian_currency(amount)
        
            display_df['Amount'] = display_df.apply(format_amount_for_display, axis=1)
        
            # 3. Round Quantity if unit is number-based
            def format_qty(row):
                unit = row['Unit']
                qty = row['Quantity']
                if VALIDATOR.is_number_based_unit(unit):
                    return int(round(qty))
                return qty
        
            display_df['Quantity'] = display_df.apply(format_qty, axis=1)
        
            # 4. Format Weighted Averages for rate mode
            if 'Weighted_Avg_Min' in display_df.columns:
                def format_weighted_avg(row, col_name):
                    val = row.get(col_name)
                    if pd.isna(val) or val is None or val == 'NA':
                        return 'NA'
                    # Always format as currency for rate mode
                    return VALIDATOR.format_indian_currency(val)
                
                display_df['Weighted_Avg_Min'] = display_df.apply(
                    lambda row: format_weighted_avg(row, 'Weighted_Avg_Min'), axis=1
                )
                display_df['Weighted_Avg_Max'] = display_df.apply(
                    lambda row: format_weighted_avg(row, 'Weighted_Avg_Max'), axis=1
                )
        
            # 5. Format Thresholds as clickable elements with file information
            if 'Threshold_Min' in display_df.columns and 'Threshold_Max' in display_df.columns:
                def format_threshold_clickable(row, col_name, files_col_name):
                    val = row.get(col_name)
                    if val == 'NA' or pd.isna(val):
                        return 'NA'
                    
                    # Format as currency
                    display_val = VALIDATOR.format_indian_currency(val)
                    
                    # Get file information
                    files_data = row.get(files_col_name, [])
                    if not files_data or len(files_data) == 0:
                        return str(display_val)
                    
                    # Create tooltip text with file information
                    tooltip_lines = []
                    for i, file_info in enumerate(files_data[:5]):
                        file_name = file_info.get('file', 'Unknown')
                        particular = file_info.get('original_particular', 'N/A')
                        tooltip_lines.append(f"{i+1}. {file_name}: {particular}")
                    
                    if len(files_data) > 5:
                        tooltip_lines.append(f"... and {len(files_data) - 5} more files")
                    
                    tooltip_text = "\\n".join(tooltip_lines)
                    tooltip_text = html.escape(tooltip_text)
                    
                    return f'<span class="threshold-clickable" title="{tooltip_text}" style="cursor: pointer; color: #007bff; text-decoration: underline;">{display_val}</span>'
                
                display_df['Threshold_Min'] = display_df.apply(
                    lambda row: format_threshold_clickable(row, 'Threshold_Min', 'Min_Threshold_Files'), axis=1
                )
                display_df['Threshold_Max'] = display_df.apply(
                    lambda row: format_threshold_clickable(row, 'Threshold_Max', 'Max_Threshold_Files'), axis=1
                )
            
        elif mode == 'amount':
            # Store raw numeric values BEFORE formatting
            display_df['Amount_Raw'] = display_df['Amount'].copy()
            
            # Format the Main Amount Column
            def format_amount_for_display(row):
                unit = row['Unit']
                amount = row['Amount']
                if VALIDATOR.is_number_based_unit(unit):
                    val = int(round(amount))
                else:
                    val = amount
                return VALIDATOR.format_indian_currency(val)
            
            display_df['Amount'] = display_df.apply(format_amount_for_display, axis=1)

            # Format Weighted Averages for amount mode
            if 'Weighted_Avg_Min' in display_df.columns:
                def format_weighted_avg(row, col_name):
                    val = row.get(col_name)
                    if pd.isna(val) or val is None or val == 'NA':
                        return 'NA'
                    # Always format as currency for amount mode
                    return VALIDATOR.format_indian_currency(val)
                
                display_df['Weighted_Avg_Min'] = display_df.apply(
                    lambda row: format_weighted_avg(row, 'Weighted_Avg_Min'), axis=1
                )
                display_df['Weighted_Avg_Max'] = display_df.apply(
                    lambda row: format_weighted_avg(row, 'Weighted_Avg_Max'), axis=1
                )

            # Format Thresholds as clickable elements with file information
            if 'Threshold_Min' in display_df.columns and 'Threshold_Max' in display_df.columns:
                def format_threshold_clickable(row, col_name, files_col_name):
                    val = row.get(col_name)
                    if val == 'NA' or pd.isna(val):
                        return 'NA'
                    
                    # Format as currency
                    display_val = VALIDATOR.format_indian_currency(val)
                    
                    # Get file information
                    files_data = row.get(files_col_name, [])
                    if not files_data or len(files_data) == 0:
                        return str(display_val)
                    
                    # Create tooltip text with file information
                    tooltip_lines = []
                    for i, file_info in enumerate(files_data[:5]):
                        file_name = file_info.get('file', 'Unknown')
                        particular = file_info.get('original_particular', 'N/A')
                        tooltip_lines.append(f"{i+1}. {file_name}: {particular}")
                    
                    if len(files_data) > 5:
                        tooltip_lines.append(f"... and {len(files_data) - 5} more files")
                    
                    tooltip_text = "&#10;".join(tooltip_lines)
                    tooltip_text = html.escape(tooltip_text)
                    
                    return f'<span class="threshold-clickable" title="{tooltip_text}" style="cursor: pointer; color: #007bff; text-decoration: underline;">{display_val}</span>'
                
                display_df['Threshold_Min'] = display_df.apply(
                    lambda row: format_threshold_clickable(row, 'Threshold_Min', 'Min_Threshold_Files'), axis=1
                )
                display_df['Threshold_Max'] = display_df.apply(
                    lambda row: format_threshold_clickable(row, 'Threshold_Max', 'Max_Threshold_Files'), axis=1
                )
            
        else:  # quantity mode
            def format_quantity_for_display(row):
                unit = row['Unit']
                quantity = row['Quantity']
                if VALIDATOR.is_number_based_unit(unit):
                    return int(round(quantity))
                return quantity
            
            display_df['Quantity'] = display_df.apply(format_quantity_for_display, axis=1)
            
            # Format Weighted Averages for quantity mode
            if 'Weighted_Avg_Min' in display_df.columns:
                def format_weighted_avg(row, col_name):
                    val = row.get(col_name)
                    if pd.isna(val) or val is None or val == 'NA':
                        return 'NA'
                    unit = row.get('Unit', '')
                    if VALIDATOR.is_number_based_unit(unit):
                        return int(round(val))
                    return f"{val:.2f}"
                
                display_df['Weighted_Avg_Min'] = display_df.apply(
                    lambda row: format_weighted_avg(row, 'Weighted_Avg_Min'), axis=1
                )
                display_df['Weighted_Avg_Max'] = display_df.apply(
                    lambda row: format_weighted_avg(row, 'Weighted_Avg_Max'), axis=1
                )
            
            # Format Thresholds as clickable elements with file information
            if 'Threshold_Min' in display_df.columns and 'Threshold_Max' in display_df.columns:
                def format_threshold_clickable(row, col_name, files_col_name):
                    val = row.get(col_name)
                    if val == 'NA' or pd.isna(val):
                        return 'NA'
                    
                    unit = row.get('Unit', '')
                    if VALIDATOR.is_number_based_unit(unit):
                        display_val = int(round(float(val)))
                    else:
                        display_val = f"{float(val):.2f}"
                    
                    # Get file information
                    files_data = row.get(files_col_name, [])
                    if not files_data or len(files_data) == 0:
                        return str(display_val)
                    
                    # Create tooltip text with file information
                    tooltip_lines = []
                    for i, file_info in enumerate(files_data[:5]):
                        file_name = file_info.get('file', 'Unknown')
                        particular = file_info.get('original_particular', 'N/A')
                        tooltip_lines.append(f"{i+1}. {file_name}: {particular}")
                    
                    if len(files_data) > 5:
                        tooltip_lines.append(f"... and {len(files_data) - 5} more files")
                    
                    tooltip_text = "&#10;".join(tooltip_lines)
                    tooltip_text = html.escape(tooltip_text)
                    
                    return f'<span class="threshold-clickable" title="{tooltip_text}" style="cursor: pointer; color: #007bff; text-decoration: underline;">{display_val}</span>'
                
                display_df['Threshold_Min'] = display_df.apply(
                    lambda row: format_threshold_clickable(row, 'Threshold_Min', 'Min_Threshold_Files'), axis=1
                )
                display_df['Threshold_Max'] = display_df.apply(
                    lambda row: format_threshold_clickable(row, 'Threshold_Max', 'Max_Threshold_Files'), axis=1
                )
        
        # Add hover tooltip for Original_Particular column
        if 'Original_Particular' in display_df.columns:
            display_df['Original_Particular'] = display_df['Original_Particular'].apply(
                lambda x: f'<span title="{html.escape(str(x))}">{html.escape(str(x))}</span>'
            )
        
        # Analysis button - not needed for grouped mode
        if mode != 'grouped':
            def create_analysis_button(row):
                if not row['Is_Valid'] and row['Threshold_Min'] != 'NA':
                    safe_particular = html.escape(row["Original_Particular"])
                    # Get the RAW value based on mode
                    if mode == 'rate':
                        value = row.get('Rate_Raw', 0)
                    elif mode == 'amount':
                        value = row.get('Amount_Raw', 0)
                    else:
                        value = row.get('Quantity', 0)
                    
                    return (f'<button class="btn btn-sm btn-info btn-analysis" '
                            f'data-particular="{safe_particular}" '
                            f'data-quantity="{value}" '
                            f'data-range="{row["Sq_Feet_Range"]}" '
                            f'data-client="{client_name}" '
                            f'data-mode="{mode}">'
                            f'View Analysis</button>')
                return ''
            
            display_df['Analysis'] = display_df.apply(create_analysis_button, axis=1)
        
        display_df['Is_Valid'] = display_df['Is_Valid'].apply(lambda x: '✅ Valid' if x else '❌ Invalid')
        
        # Dynamic column order based on mode
        if mode == 'grouped':
            column_order = ['Sr.No', 'Particulars', 'As_Per_CWI_Amount', 'Rate_per_sqfeet',
                           'Sq_Feet_Range', 'Is_Valid',
                           'CWI_Validation', 'Rate_Validation',
                           'CWI_Weighted_Min', 'CWI_Weighted_Max',  # ✅ NEW
                           'Rate_Weighted_Min', 'Rate_Weighted_Max']  # ✅ NEW
        elif mode == 'amount':
            column_order = ['Sr.No', 'Original_Particular', 'Amount', 'Unit', 
                           'Sq_Feet_Range', 'Is_Valid', 'Validation_Message', 
                           'Matched_Particular', 'Similarity_Score', 
                           'Threshold_Min', 'Threshold_Max',
                           'Weighted_Avg_Min', 'Weighted_Avg_Max',
                           'Analysis']
        elif mode == 'rate':
            if is_regional:
                column_order = ['Sr.No', 'Original_Particular', 'Rate', 'Unit', 'Quantity', 'Amount',
                               'Sq_Feet_Range', 'Is_Valid', 
                               'Validation_Message_East', 'Validation_Message_West', 
                               'Validation_Message_North', 'Validation_Message_South',
                               'Matched_Particular', 'Similarity_Score', 
                               'Threshold_Min', 'Threshold_Max',
                               'Weighted_Avg_Min', 'Weighted_Avg_Max',
                               'Analysis']
            else:
                column_order = ['Sr.No', 'Original_Particular', 'Rate', 'Unit', 'Quantity', 'Amount',
                               'Sq_Feet_Range', 'Is_Valid', 'Validation_Message', 
                               'Matched_Particular', 'Similarity_Score', 
                               'Threshold_Min', 'Threshold_Max',
                               'Weighted_Avg_Min', 'Weighted_Avg_Max',
                           'Analysis']
        else:  # quantity mode
            column_order = ['Sr.No', 'Original_Particular', 'Quantity', 'Unit', 
                           'Sq_Feet_Range', 'Is_Valid', 'Validation_Message', 
                           'Matched_Particular', 'Similarity_Score', 
                           'Threshold_Min', 'Threshold_Max', 
                           'Weighted_Avg_Min', 'Weighted_Avg_Max',
                           'Analysis']
        
        display_df = display_df[column_order]
        
        # Generate HTML table
        results_html = display_df.to_html(classes='table table-striped table-hover', justify='left', index=False, escape=False)

        # ✅ FIX: Add CSS class to selected region column if regional mode
        if is_regional and selected_region:
            region_col_name = f'Validation_Message_{selected_region.capitalize()}'

            # Add green header class to selected region
            results_html = results_html.replace(
                f'<th>{region_col_name}</th>',
                f'<th class="selected-region-header">{region_col_name} ✓</th>'
            )
    
   
        
        total = len(results_df)
        valid = results_df['Is_Valid'].sum()
        invalid = total - valid
        accuracy = (valid / total * 100) if total > 0 else 0
        
        print(f"📈 Statistics: Total={total}, Valid={valid}, Invalid={invalid}")
        print(f"✅ Rendering results with mode: {mode}")
        print("="*60 + "\n")
        
        return render_template('results.html', 
                table_html=results_html, 
                full_report_filename=full_report_filename,
                invalid_report_filename=invalid_report_filename,
                total=total, valid=valid, invalid=invalid, 
                accuracy=f"{accuracy:.2f}%",
                client=client_name,
                validation_mode=mode,
                region=selected_region)
                
    except Exception as e:
        print(f"❌ Error in switch_mode: {e}")
        import traceback
        traceback.print_exc()
        flash(f'Error switching mode: {str(e)}')
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)