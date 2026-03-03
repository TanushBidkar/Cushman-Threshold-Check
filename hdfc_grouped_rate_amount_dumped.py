import pandas as pd
import numpy as np
import os
import pickle
import re
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

def _default_dict_factory():
    """Factory function for nested defaultdict"""
    return defaultdict(list)

class CWIGroupedValidator:
    def __init__(self):
        # Dictionaries to store data with DUAL RANGES
        
        # 1. Rate Data (250 sqft buckets) - From Summary Sheet
        self.rate_sqfeet_ranges = defaultdict(lambda: {
            'grouped_ranges': {}, 
            'particulars_data': defaultdict(list), 
            'subcategory_data': defaultdict(_default_dict_factory)
        })
        
        # 2. Amount As per CWI (500 sqft buckets) - From Summary Sheet
        self.amount_cwi_ranges = defaultdict(lambda: {
            'grouped_ranges': {}, 
            'particulars_data': defaultdict(list)
        })
        
        # 3. Rate per sq.feet Data - 250 sqft (From Extracted Data)
        self.rate_per_sqft_250_ranges = defaultdict(lambda: {
            'grouped_ranges': {}, 
            'subcategory_data': defaultdict(_default_dict_factory)
        })
        
        # 4. As per CWI Amount - 250 sqft (From Extracted Data)
        self.amount_per_cwi_250_ranges = defaultdict(lambda: {
            'grouped_ranges': {}, 
            'subcategory_data': defaultdict(_default_dict_factory)
        })
        
        self.all_groups = set()
        
        # Mapping Storage
        self.all_subcategories = defaultdict(set) 
        self.subcategory_to_group = {} 
        self.subcategory_particulars = defaultdict(set)
        
        # STRICT Standard Groups Configuration (HDFC)
        self.known_groups = {
            'CIVIL & RELATED WORKS': ['civil', 'civil &', 'civil and', 'related work'],
            'POP & FALSE CEILING WORKS': ['pop', 'false ceiling', 'pop &', 'pop and'],
            'CARPENTRY AND INTERIOR WORKS': ['carpentry', 'carpentary', 'carpenter', 'interior', 'carpentory', 'wood work'],
            'PAINTING WORKS': ['painting', 'paint'],
            'ROLLING SHUTTER AND MS WORK': ['rolling shutter', 'rolling shutters', 'ms work', 'ms works', 'grill'],
            'ELECTRIFICATION AND ALLIED WORKS': ['electrification', 'electrificaton', 'electric', 'allied work', 'electrical'],
            'ADDITIONAL WORKS': ['additional', 'addi', 'additonal'],
            'TOTAL': ['total', 'grand total']  # Added TOTAL group
        }
        
        # The catch-all bucket
        self.fallback_group = 'Others/Uncategorized'

    def clean_text_for_matching(self, text):
        """Normalize text ONLY for keyword matching - NOT for storage"""
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    def normalize_subcategory(self, text):
        """
        Smart normalization that preserves original text but allows matching:
        - Keeps original capitalization and spelling
        - Only fixes common variations (work/works, single/double letters)
        - Preserves special characters if present
        """
        if pd.isna(text) or str(text).strip() == '' or str(text).lower() == 'nan': 
            return None
        
        # Get original text stripped
        original = str(text).strip()
        
        return original  # Return ORIGINAL text, not modified

    def create_lookup_key(self, text):
        """Create a standardized key for matching while preserving original text"""
        if pd.isna(text) or str(text).strip() == '':
            return None
        
        key = str(text).lower().strip()
        
        # Normalize variations
        key = re.sub(r'\bworks?\b', 'work', key)  # work/works
        key = re.sub(r'll', 'l', key)  # panelling/paneling
        key = re.sub(r'([a-z])\1+', r'\1', key)  # double letters (but keep some)
        key = ' '.join(key.split())  # normalize spaces
        
        return key

    def extract_sqfeet_from_filename(self, filename):
        """
        Smartly extracts sqft and returns TWO different range strings:
        1. Rate Range (250 sqft intervals)
        2. Amount Range (500 sqft intervals)
        """
        filename_clean = filename.lower().replace('_', ' ').replace('-', ' ')
        
        # Regex to find digits immediately followed by sq, sqft, sq.ft, or sq feet
        match = re.search(r'(\d+)\s*(?:sq|sft|sqft|sq\.ft|sq\.feet)', filename_clean)
        
        sqft_val = None
        if match:
            try: sqft_val = int(match.group(1))
            except ValueError: pass
        
        # Fallback: Look for large numbers at start if explicit 'sqft' missing
        if sqft_val is None:
            start_match = re.match(r'^(\d{3,5})', filename)
            if start_match:
                try:
                    val = int(start_match.group(1))
                    if 200 < val < 30000: # Reasonable range check
                        sqft_val = val
                except: pass
                
        if sqft_val is None or sqft_val == 0:
            return None, None, None

        # 1. Calculate Rate Range (250 sqft intervals)
        rate_lower = ((sqft_val - 1) // 250) * 250 + 1
        rate_upper = rate_lower + 249
        rate_range_str = f"{rate_lower}-{rate_upper}"
        
        # 2. Calculate Amount Range (250 sqft intervals)
        amount_lower = ((sqft_val - 1) // 250) * 250 + 1
        amount_upper = amount_lower + 249
        amount_range_str = f"{amount_lower}-{amount_upper}"
        
        return rate_range_str, amount_range_str, sqft_val

    def standardize_group_name(self, particular_text):
        """Map extracted text to a Standard Group Name or Fallback"""
        if pd.isna(particular_text) or str(particular_text).strip() == '': return None
        cleaned_text = self.clean_text_for_matching(particular_text)
        
        # 1. Check Known Groups
        for standard_name, keywords in self.known_groups.items():
            for keyword in keywords:
                if keyword in cleaned_text:
                    self.all_groups.add(standard_name)
                    return standard_name
        
        # 2. If not found, strictly categorize as Others
        self.all_groups.add(self.fallback_group)
        return self.fallback_group

    def extract_amount(self, value):
        """Clean currency strings into floats"""
        if pd.isna(value): return 0.0
        s = str(value).replace(',', '').replace('₹', '').replace('Rs', '').replace('Rs.', '').strip()
        s = re.sub(r'[^\d\.\-]', '', s) # Remove non-numeric except dot and minus
        try: return float(s)
        except: return 0.0

    def calculate_stats_with_file_tracking(self, data_list):
        """
        Calculates: Mean, Weighted Range (±1SD)
        AND tracks which file had the Actual Min/Max.
        """
        # Extract pure values for math
        values = np.array([item['val'] for item in data_list])
        non_zero_indices = [i for i, x in enumerate(values) if x != 0]
        non_zero_values = values[non_zero_indices]
        
        stats = {
            'total_count': len(values),
            'zero_count': len(values) - len(non_zero_values),
            'mean': 0.0,
            'std': 0.0,
            
            # Weighted Range (Mean ± 1SD)
            'weighted_min': 0.0, 'weighted_max': 0.0,
            
            # File Tracking (Actual Min/Max found in files)
            'abs_min_val': 0.0, 'abs_min_file': '', 'abs_min_part': '',
            'abs_max_val': 0.0, 'abs_max_file': '', 'abs_max_part': ''
        }
        
        if len(non_zero_values) > 0:
            # 1. Get Actual Min/Max FIRST
            actual_min = np.min(non_zero_values)
            actual_max = np.max(non_zero_values)

            # 2. Math Stats
            mean_val = np.mean(non_zero_values)
            std_val = np.std(non_zero_values)
            
            stats['mean'] = mean_val
            stats['std'] = std_val
            
            # 3. Weighted Average Range (Stricter: Mean +/- 1 SD)
            # FIX: Clamp the lower bound so it cannot be lower than the Actual Min
            calculated_min = mean_val - std_val
            stats['weighted_min'] = max(actual_min, calculated_min)
            stats['weighted_max'] = mean_val + std_val
            
            # 4. Absolute Min/Max Tracking (The "Actual" Thresholds)
            valid_items = [data_list[i] for i in non_zero_indices]
            
            # Max
            max_item = max(valid_items, key=lambda x: x['val'])
            stats['abs_max_val'] = max_item['val']
            stats['abs_max_file'] = max_item['file']
            stats['abs_max_part'] = max_item['particular']
            
            # Min
            min_item = min(valid_items, key=lambda x: x['val'])
            stats['abs_min_val'] = min_item['val']
            stats['abs_min_file'] = min_item['file']
            stats['abs_min_part'] = min_item['particular']

        return stats

    def load_training_data(self, folder_path):
        print(f"Loading files from: {folder_path}")
        try:
            files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]
        except FileNotFoundError:
            print(f"Error: Folder not found: {folder_path}")
            return []

        if not files:
            print("No Excel files found.")
            return []
            
        training_records = []
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            
            # --- STEP 1: SMART FILENAME EXTRACTION ---
            rate_range_str, amount_range_str, extracted_sqft = self.extract_sqfeet_from_filename(file)
            
            print(f"\nProcessing: {file}")
            if not extracted_sqft:
                print(f"  ⚠ WARNING: Could not detect sqft. Skipping.")
                continue
            
            print(f"  ✓ SqFt: {extracted_sqft} | 250 Range: {rate_range_str} | 500 Range: {amount_range_str}")
            
            # --- STEP 2: SUMMARY SHEET (Main Groups) - NOW WITH BOTH RATE AND AMOUNT ---
            try:
                df_summary = pd.read_excel(file_path, sheet_name='Summary Sheet')
                df_summary.columns = [str(c).strip() for c in df_summary.columns]
                
                col_part = None
                col_rate = None
                col_amount_cwi = None
                
                for col in df_summary.columns:
                    c_lower = col.lower()
                    
                    # Find Particular
                    if 'particular' in c_lower: 
                        col_part = col
                    
                    # Find Rate per sq.feet
                    if 'rate' in c_lower and 'sq' in c_lower: 
                        col_rate = col
                    
                    # Find Amount As per CWI
                    if 'amount' in c_lower and 'cwi' in c_lower:
                        col_amount_cwi = col
                
                # Fallback for CWI Amount
                if not col_amount_cwi:
                    for col in df_summary.columns:
                        if 'cwi' in col.lower() and 'qty' not in col.lower(): 
                            col_amount_cwi = col
                            break
                            
                if col_part and col_rate:
                    # NOTE: NOT Filtering 'Total' here so we can capture the TOTAL group
                    
                    for _, row in df_summary.iterrows():
                        p_name = str(row[col_part]).strip()  # PRESERVE ORIGINAL EXACTLY
                        
                        # Skip if empty or nan
                        if pd.isna(p_name) or p_name == '' or p_name.lower() == 'nan':
                            continue

                        rt = self.extract_amount(row[col_rate])
                        amt_cwi = self.extract_amount(row[col_amount_cwi]) if col_amount_cwi else 0.0
                        
                        group = self.standardize_group_name(p_name)
                        
                        if group:
                            # Add RATE data to 250 sqft buckets
                            if rt > 0:
                                self.rate_sqfeet_ranges[rate_range_str]['particulars_data'][group].append({
                                    'val': rt, 'file': file, 'particular': p_name
                                })
                            
                            # Add AMOUNT AS PER CWI data to 500 sqft buckets
                            if amt_cwi > 0:
                                self.amount_cwi_ranges[amount_range_str]['particulars_data'][group].append({
                                    'val': amt_cwi, 'file': file, 'particular': p_name
                                })
                            
                            if rt > 0 or amt_cwi > 0:
                                training_records.append({'file': file, 'group': group, 'sheet': 'Summary'})
                            
                    print(f"  ✓ Summary Sheet processed (Rate + Amount CWI including TOTAL)")
                else:
                    print(f"  ⚠ Missing columns in Summary Sheet")
            except Exception as e:
                print(f"  ✗ Error reading Summary Sheet: {e}")

            # --- STEP 3: EXTRACTED DATA SHEET (Subcategories) - DUAL COLUMNS ---
            try:
                df_ext = pd.read_excel(file_path, sheet_name='Extracted Data')
                col_map = {str(c).lower().strip(): c for c in df_ext.columns}
                
                # Find required columns
                key_sub = None
                for k in col_map:
                    if 'sub' in k and 'cat' in k: 
                        key_sub = k
                        break

                key_nature = next((k for k in col_map if 'nature' in k), None)
                key_particular = next((k for k in col_map if 'particular' in k), None)
                
                # Find "Rate per sq.feet" column
                key_rate_per_sqft = None
                for k in col_map:
                    if 'rate' in k and 'sq' in k and 'feet' in k:
                        key_rate_per_sqft = k
                        break
                
                # Find "As per CWI (Amount)" column
                key_amount_cwi = None
                for k in col_map:
                    if 'as per cwi' in k and 'amount' in k:
                        key_amount_cwi = k
                        break
                
                if not key_amount_cwi:
                    # Fallback
                    for k in col_map:
                        if 'cwi' in k and 'amount' in k:
                            key_amount_cwi = k
                            break
                
                if key_sub and key_nature and (key_rate_per_sqft or key_amount_cwi):
                    print(f"  ✓ Found Extracted Data columns:")
                    print(f"    - Sub Category: {key_sub}")
                    print(f"    - Nature: {key_nature}")
                    if key_rate_per_sqft:
                        print(f"    - Rate per sq.feet: {key_rate_per_sqft}")
                    if key_amount_cwi:
                        print(f"    - As per CWI Amount: {key_amount_cwi}")
                    
                    # AGGREGATION LOGIC with LOOKUP KEYS
                    # Use lookup keys for grouping, but store ORIGINAL text
                    subcategory_aggregator = defaultdict(lambda: {
                        'original_name': None,
                        'rates': [], 
                        'amounts_cwi': [],
                        'nature_samples': [],
                        'particulars': []
                    })
                    
                    for _, row in df_ext.iterrows():
                        sub_cat_raw = str(row[col_map[key_sub]]).strip()
                        nature = str(row[col_map[key_nature]]).strip()
                        particular = str(row[col_map[key_particular]]).strip() if key_particular else nature
                        
                        rate_sqft = self.extract_amount(row[col_map[key_rate_per_sqft]]) if key_rate_per_sqft else 0.0
                        amount_cwi = self.extract_amount(row[col_map[key_amount_cwi]]) if key_amount_cwi else 0.0
                        
                        if pd.isna(nature) or nature == '' or nature.lower() == 'nan': 
                            continue
                        
                        # Create lookup key for grouping
                        lookup_key = self.create_lookup_key(sub_cat_raw)
                        
                        if lookup_key and (rate_sqft > 0 or amount_cwi > 0):
                            # Store ORIGINAL name (first occurrence)
                            if subcategory_aggregator[lookup_key]['original_name'] is None:
                                subcategory_aggregator[lookup_key]['original_name'] = sub_cat_raw
                            
                            if rate_sqft > 0:
                                subcategory_aggregator[lookup_key]['rates'].append(rate_sqft)
                            if amount_cwi > 0:
                                subcategory_aggregator[lookup_key]['amounts_cwi'].append(amount_cwi)
                            
                            subcategory_aggregator[lookup_key]['nature_samples'].append(nature)
                            subcategory_aggregator[lookup_key]['particulars'].append(particular)
                    
                    # Now aggregate and store
                    for lookup_key, data in subcategory_aggregator.items():
                        original_name = data['original_name']  # USE ORIGINAL NAME
                        total_rate = sum(data['rates'])
                        total_amount_cwi = sum(data['amounts_cwi'])
                        count_rate = len(data['rates'])
                        count_amount = len(data['amounts_cwi'])
                        
                        # Find group from nature of works (use first sample)
                        group = self.standardize_group_name(data['nature_samples'][0])
                        
                        if group:
                            # Link Subcat to Group (using ORIGINAL name)
                            self.all_subcategories[group].add(original_name)
                            self.subcategory_to_group[original_name] = group
                            for nature in data['nature_samples']:
                                self.subcategory_particulars[original_name].add(nature)
                            
                            # Store RATE data in 250 sqft ranges
                            if total_rate > 0:
                                agg_info_rate = f"{original_name} (aggregated: {count_rate} items, total={total_rate:.2f})"
                                self.rate_per_sqft_250_ranges[rate_range_str]['subcategory_data'][group][original_name].append({
                                    'val': total_rate, 
                                    'file': file, 
                                    'particular': agg_info_rate
                                })
                                print(f"    → [250] Aggregated Rate: {original_name} = {total_rate:.2f} ({count_rate} items)")
                            
                            # Store AMOUNT CWI data in 500 sqft ranges
                            if total_amount_cwi > 0:
                                agg_info_amount = f"{original_name} (aggregated: {count_amount} items, total={total_amount_cwi:.2f})"
                                self.amount_per_cwi_250_ranges[amount_range_str]['subcategory_data'][group][original_name].append({
                                    'val': total_amount_cwi,
                                    'file': file,
                                    'particular': agg_info_amount
                                })
                                print(f"    → [250] Aggregated Amount CWI: {original_name} = {total_amount_cwi:.2f} ({count_amount} items)")
                    
                    print(f"  ✓ Extracted Data processed: {len(subcategory_aggregator)} subcategories")
                else:
                    missing = []
                    if not key_sub: missing.append("Sub Category")
                    if not key_nature: missing.append("Nature of Works")
                    if not key_rate_per_sqft and not key_amount_cwi: 
                        missing.append("Rate per sq.feet OR As per CWI Amount")
                    print(f"  ⚠ Missing columns in Extracted Data: {', '.join(missing)}")
                            
            except Exception as e:
                print(f"  ✗ Error reading Extracted Data: {e}")
                
        return training_records

    def process_and_train(self):
        """Calculate AND PRINT statistics for every range, group, and subcategory"""
        print("\n" + "="*80)
        print("CALCULATING STATS FOR HDFC (MEAN, WEIGHTED RANGES & FILE TRACKING)")
        print("="*80)
        
        # 1. SUMMARY SHEET RATES (250 sqft intervals)
        print("\n>>> SECTION 1: SUMMARY SHEET - RATE PER SQ.FEET (250 sqft intervals) <<<")
        sorted_rate_ranges = sorted(self.rate_sqfeet_ranges.keys(), key=lambda x: int(x.split('-')[0]))
        
        for rng in sorted_rate_ranges:
            data = self.rate_sqfeet_ranges[rng]
            print(f"\n[ RATE RANGE: {rng} ]")
            
            # Main Groups from Summary Sheet
            for group, values in data['particulars_data'].items():
                stats = self.calculate_stats_with_file_tracking(values)
                data['grouped_ranges'][group] = {'rate_stats': stats}
                
                if stats['mean'] > 0:
                    print(f"  • {group}: Mean={stats['mean']:.2f}")
                    print(f"    Weighted Range (±1SD): {stats['weighted_min']:.2f} - {stats['weighted_max']:.2f}")
                    print(f"    [ACTUAL MIN]: {stats['abs_min_val']:.2f} (File: {stats['abs_min_file']})")
                    print(f"    [ACTUAL MAX]: {stats['abs_max_val']:.2f} (File: {stats['abs_max_file']})")

        # 2. SUMMARY SHEET AMOUNT AS PER CWI (500 sqft intervals)
        print("\n>>> SECTION 2: SUMMARY SHEET - AMOUNT AS PER CWI (500 sqft intervals) <<<")
        sorted_amount_ranges = sorted(self.amount_cwi_ranges.keys(), key=lambda x: int(x.split('-')[0]))
        
        for rng in sorted_amount_ranges:
            data = self.amount_cwi_ranges[rng]
            print(f"\n[ AMOUNT RANGE: {rng} ]")
            
            # Main Groups from Summary Sheet
            for group, values in data['particulars_data'].items():
                stats = self.calculate_stats_with_file_tracking(values)
                data['grouped_ranges'][group] = {'amount_stats': stats}
                
                if stats['mean'] > 0:
                    print(f"  • {group}: Mean={stats['mean']:.2f}")
                    print(f"    Weighted Range (±1SD): {stats['weighted_min']:.2f} - {stats['weighted_max']:.2f}")
                    print(f"    [ACTUAL MIN]: {stats['abs_min_val']:.2f} (File: {stats['abs_min_file']})")
                    print(f"    [ACTUAL MAX]: {stats['abs_max_val']:.2f} (File: {stats['abs_max_file']})")

        # 3. EXTRACTED DATA - RATE PER SQ.FEET (250 sqft intervals)
        print("\n>>> SECTION 3: EXTRACTED DATA - RATE PER SQ.FEET (250 sqft intervals) <<<")
        sorted_250_ranges = sorted(self.rate_per_sqft_250_ranges.keys(), key=lambda x: int(x.split('-')[0]))
        
        for rng in sorted_250_ranges:
            data = self.rate_per_sqft_250_ranges[rng]
            print(f"\n[ 250 RANGE: {rng} ]")
            
            # Process subcategories
            for group, sub_data in data['subcategory_data'].items():
                if group not in data['grouped_ranges']:
                    data['grouped_ranges'][group] = {'sub_stats': {}}
                
                print(f"  GROUP: {group}")
                for sub_cat, sub_values in sub_data.items():
                    sub_stats = self.calculate_stats_with_file_tracking(sub_values)
                    data['grouped_ranges'][group]['sub_stats'][sub_cat] = sub_stats
                    
                    if sub_stats['mean'] > 0:
                        print(f"    └─ Subcategory: {sub_cat}")
                        print(f"       Mean: {sub_stats['mean']:.2f}")
                        print(f"       Weighted Range (±1SD): {sub_stats['weighted_min']:.2f} - {sub_stats['weighted_max']:.2f}")
                        print(f"       [ACTUAL MIN]: {sub_stats['abs_min_val']:.2f} | File: {sub_stats['abs_min_file']}")
                        print(f"       [ACTUAL MAX]: {sub_stats['abs_max_val']:.2f} | File: {sub_stats['abs_max_file']}")

        # 4. EXTRACTED DATA - AMOUNT AS PER CWI (250 sqft intervals)
        print("\n>>> SECTION 4: EXTRACTED DATA - AMOUNT AS PER CWI (250 sqft intervals) <<<")
        sorted_250_amt_ranges = sorted(self.amount_per_cwi_250_ranges.keys(), key=lambda x: int(x.split('-')[0]))
        
        for rng in sorted_250_amt_ranges:
            data = self.amount_per_cwi_250_ranges[rng]
            print(f"\n[ 250 RANGE: {rng} ]")
            
            # Process subcategories
            for group, sub_data in data['subcategory_data'].items():
                if group not in data['grouped_ranges']:
                    data['grouped_ranges'][group] = {'sub_stats': {}}
                
                print(f"  GROUP: {group}")
                for sub_cat, sub_values in sub_data.items():
                    sub_stats = self.calculate_stats_with_file_tracking(sub_values)
                    data['grouped_ranges'][group]['sub_stats'][sub_cat] = sub_stats
                    
                    if sub_stats['mean'] > 0:
                        print(f"    └─ Subcategory: {sub_cat}")
                        print(f"       Mean: {sub_stats['mean']:.2f}")
                        print(f"       Weighted Range (±1SD): {sub_stats['weighted_min']:.2f} - {sub_stats['weighted_max']:.2f}")
                        print(f"       [ACTUAL MIN]: {sub_stats['abs_min_val']:.2f} | File: {sub_stats['abs_min_file']}")
                        print(f"       [ACTUAL MAX]: {sub_stats['abs_max_val']:.2f} | File: {sub_stats['abs_max_file']}")

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)

    def save_model(self, output_folder):
        """Save the trained object and generate report"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        path = os.path.join(output_folder, "trained_model_HDFC_grouped_rate_amount_dumped_South.pkl")
        
        model_payload = {
            'rate_sqfeet_ranges': dict(self.rate_sqfeet_ranges),
            'amount_cwi_ranges': dict(self.amount_cwi_ranges),
            'rate_per_sqft_250_ranges': dict(self.rate_per_sqft_250_ranges),
            'amount_per_cwi_250_ranges': dict(self.amount_per_cwi_250_ranges),
            'known_groups': self.known_groups,
            'fallback_group': self.fallback_group,
            'subcategory_map': dict(self.all_subcategories),
            'version': 'HDFC_V5_Rate_And_Amount_With_Total_Smart_Matching'
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_payload, f)
        
        print(f"\n✓ Model Saved Successfully to: {path}")
        
        # Generate Detailed Text Report
        self.generate_report(output_folder)

    def generate_report(self, output_folder):
        report_path = os.path.join(output_folder, "Final_Training_Report_HDFC_Complete.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HDFC FINAL VALIDATION MODEL REPORT - COMPLETE\n")
            f.write("="*80 + "\n\n")
        
            # SECTION 1: SUMMARY SHEET RATES (250)
            f.write("SECTION 1: SUMMARY SHEET - RATE PER SQ.FEET (250 sqft intervals)\n")
            f.write("-" * 80 + "\n")
            sorted_rate_ranges = sorted(self.rate_sqfeet_ranges.keys(), key=lambda x: int(x.split('-')[0]))
            
            for rng in sorted_rate_ranges:
                f.write(f"\n[ RANGE: {rng} sq.ft ]\n")
                data = self.rate_sqfeet_ranges[rng]['grouped_ranges']
                for group, stats in data.items():
                    r_stats = stats.get('rate_stats')
                    if r_stats and r_stats['mean'] > 0:
                        f.write(f"  > {group}\n")
                        f.write(f"    Mean: {r_stats['mean']:.2f}\n")
                        f.write(f"    Weighted Range: {r_stats['weighted_min']:.2f} - {r_stats['weighted_max']:.2f}\n")
                        f.write(f"    Actual Min: {r_stats['abs_min_val']:.2f} ({r_stats['abs_min_file']})\n")
                        f.write(f"    Actual Max: {r_stats['abs_max_val']:.2f} ({r_stats['abs_max_file']})\n")

            # SECTION 2: SUMMARY SHEET AMOUNT CWI (500)
            f.write("\n\n" + "="*80 + "\n")
            # SECTION 2: SUMMARY SHEET AMOUNT CWI (250)
            f.write("SECTION 2: SUMMARY SHEET - AMOUNT AS PER CWI (250 sqft intervals)\n")
            f.write("-" * 80 + "\n")
            sorted_amount_ranges = sorted(self.amount_cwi_ranges.keys(), key=lambda x: int(x.split('-')[0]))
            
            for rng in sorted_amount_ranges:
                f.write(f"\n[ RANGE: {rng} sq.ft ]\n")
                data = self.amount_cwi_ranges[rng]['grouped_ranges']
                for group, stats in data.items():
                    a_stats = stats.get('amount_stats')
                    if a_stats and a_stats['mean'] > 0:
                        f.write(f"  > {group}\n")
                        f.write(f"    Mean: {a_stats['mean']:.2f}\n")
                        f.write(f"    Weighted Range: {a_stats['weighted_min']:.2f} - {a_stats['weighted_max']:.2f}\n")
                        f.write(f"    Actual Min: {a_stats['abs_min_val']:.2f} ({a_stats['abs_min_file']})\n")
                        f.write(f"    Actual Max: {a_stats['abs_max_val']:.2f} ({a_stats['abs_max_file']})\n")

            # SECTION 3: EXTRACTED DATA - RATE (250)
            f.write("\n\n" + "="*80 + "\n")
            f.write("SECTION 3: EXTRACTED DATA - RATE PER SQ.FEET (250 sqft intervals)\n")
            f.write("-" * 80 + "\n")
            sorted_250_ranges = sorted(self.rate_per_sqft_250_ranges.keys(), key=lambda x: int(x.split('-')[0]))
            
            for rng in sorted_250_ranges:
                f.write(f"\n[ RANGE: {rng} sq.ft ]\n")
                data = self.rate_per_sqft_250_ranges[rng]['grouped_ranges']
                for group, group_data in data.items():
                    f.write(f"\n  GROUP: {group}\n")
                    sub_stats_dict = group_data.get('sub_stats', {})
                    for sub_cat, sub_stats in sub_stats_dict.items():
                        if sub_stats['mean'] > 0:
                            f.write(f"    → {sub_cat}\n")
                            f.write(f"      Mean: {sub_stats['mean']:.2f}\n")
                            f.write(f"      Weighted Range: {sub_stats['weighted_min']:.2f} - {sub_stats['weighted_max']:.2f}\n")
                            f.write(f"      Actual Min: {sub_stats['abs_min_val']:.2f} ({sub_stats['abs_min_file']})\n")
                            f.write(f"      Actual Max: {sub_stats['abs_max_val']:.2f} ({sub_stats['abs_max_file']})\n")

            # SECTION 4: EXTRACTED DATA - AMOUNT CWI (250)
            f.write("SECTION 4: EXTRACTED DATA - AMOUNT AS PER CWI (250 sqft intervals)\n")
            f.write("-" * 80 + "\n")
            sorted_250_amt_ranges = sorted(self.amount_per_cwi_250_ranges.keys(), key=lambda x: int(x.split('-')[0]))
            
            for rng in sorted_250_amt_ranges:
                f.write(f"\n[ RANGE: {rng} sq.ft ]\n")
                data = self.amount_per_cwi_250_ranges[rng]['grouped_ranges']
                for group, group_data in data.items():
                    f.write(f"\n  GROUP: {group}\n")
                    sub_stats_dict = group_data.get('sub_stats', {})
                    for sub_cat, sub_stats in sub_stats_dict.items():
                        if sub_stats['mean'] > 0:
                            f.write(f"    → {sub_cat}\n")
                            f.write(f"      Mean: {sub_stats['mean']:.2f}\n")
                            f.write(f"      Weighted Range: {sub_stats['weighted_min']:.2f} - {sub_stats['weighted_max']:.2f}\n")
                            f.write(f"      Actual Min: {sub_stats['abs_min_val']:.2f} ({sub_stats['abs_min_file']})\n")
                            f.write(f"      Actual Max: {sub_stats['abs_max_val']:.2f} ({sub_stats['abs_max_file']})\n")

        print(f"✓ Report generated at: {report_path}")

def main():
    # 1. Initialize
    validator = CWIGroupedValidator()
    # 2. Path Settings (HDFC Paths)
    # Ensure these paths are correct for your local machine
    TRAINING_FOLDER = r"C:\Users\Tanush.Bidkar\Downloads\training_files 2\training_files 2\training_files\South"
    OUTPUT_FOLDER = r"C:\Users\Tanush.Bidkar\Downloads\training_files 2\training_files 2\training_files\South"

    # 3. Load & Train
    print("="*80)
    print("Starting Training Process (HDFC - Rate AND Amount CWI with TOTAL)")
    print("="*80)
    records = validator.load_training_data(TRAINING_FOLDER)

    if len(records) > 0:
        validator.process_and_train()
        validator.save_model(OUTPUT_FOLDER)
        print("\n" + "="*80)
        print("✓ ALL PROCESSES COMPLETED SUCCESSFULLY")
        print("="*80)
    else:
        print("\n⚠ No records processed. Please check folder path and filenames.")

if __name__ == "__main__":
    main()