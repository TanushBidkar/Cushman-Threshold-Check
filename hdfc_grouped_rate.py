import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

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
        # Text similarity models for each range
        self.vectorizers = {}
        self.particular_vectors = {}
        self.particular_names = {}
        self.training_stats = {}
        self.unknown_particulars = set()
        self.all_groups = set()
        self.new_groups_found = set()
        
        # Standard grouped categories with their keywords
        self.known_groups = {
            'Civil and Related Works': ['civil', 'civil &', 'civil and', 'related work'],
            'POP & False Ceiling Works': ['pop', 'false ceiling', 'pop &', 'pop and'],
            'Carpentry and Interior Works': ['carpentry', 'carpentary', 'carpenter', 'interior', 'carpentory'],
            'Painting Works': ['painting'],
            'Rolling Shutter and MS Work': ['rolling shutter', 'rolling shutters', 'ms work', 'ms works', 'rolling', 'shutter'],
            'Electrification and Allied Works': ['electrification', 'electrificaton', 'electric', 'allied work'],
            'Accessibility Related Works': ['accessibility', 'accesibility', 'accessible', 'accessible branch'],
            'Non-Tender Items': ['non-tender', 'non tender', 'nt', 'n-t', 'n t', 'nontender'],
            'Additional Works': ['additional', 'addi', 'additonal', 'aditional'],
            'Interior Works': ['interior work', 'interiors']
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
    
    def find_cwi_column(self, df):
        """Find column containing 'CWI' keyword"""
        for col in df.columns:
            col_str = str(col).lower()
            if 'cwi' in col_str:
                return col
        return None
    
    def find_rate_column(self, df):
        """Find the last column which should be 'Rate per sq.feet'"""
        return df.columns[-1]
    
    def find_total_row_index(self, df):
        """Find the first row where 'Total' or 'Grand Total' appears in first two columns"""
        for idx, row in df.iterrows():
            first_col_val = str(row.iloc[0]).strip().lower() if not pd.isna(row.iloc[0]) else ''
            second_col_val = str(row.iloc[1]).strip().lower() if len(row) > 1 and not pd.isna(row.iloc[1]) else ''
            
            if first_col_val in ['total', 'grand total'] or second_col_val in ['total', 'grand total']:
                return idx
        
        return None

    def load_training_data(self, folder_path):
        """Load all Excel files from training folder and extract grouped data from Summary Sheet"""
        training_data = []
        
        print(f"Loading training data from: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} does not exist!")
            return training_data
        
        excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            print("No Excel files found in the training folder!")
            return training_data
        
        print(f"Found {len(excel_files)} Excel files")
        
        for file_name in excel_files:
            file_path = os.path.join(folder_path, file_name)
            print(f"\nProcessing: {file_name}")
            
            # Extract sq feet range from filename
            sqfeet_range = self.extract_sqfeet_from_filename(file_name)
            if sqfeet_range is None:
                print(f"⚠ Warning: Could not extract sq feet range from filename: {file_name}")
                print("Please ensure filename contains sq feet range (e.g., '500-750', '1001-1250', etc.)")
                continue
            
            print(f"  Detected sq feet range: {sqfeet_range}")
            
            try:
                df = pd.read_excel(file_path, sheet_name='Summary Sheet')
                
                print(f"  Sheet 'Summary Sheet' loaded with {len(df)} rows and {len(df.columns)} columns")
                
                cwi_column = self.find_cwi_column(df)
                if cwi_column is None:
                    print(f"  ⚠ Warning: Could not find column containing 'CWI' in {file_name}")
                    continue
                
                print(f"  Found CWI column: '{cwi_column}'")
                
                rate_column = self.find_rate_column(df)
                print(f"  Using last column as Rate: '{rate_column}'")
                
                total_row_idx = self.find_total_row_index(df)
                
                if total_row_idx is not None:
                    print(f"  Found 'Total' at row index {total_row_idx}, including it")
                    df = df.iloc[:total_row_idx + 1] 
                else:
                    print(f"  No 'Total' row found, processing all rows")
                
                df_extracted = pd.DataFrame({
                    'Sr.No': df.iloc[:, 0],
                    'Particulars': df.iloc[:, 1],
                    'As Per CWI': df[cwi_column],
                    'Rate per sq.feet': df[rate_column]
                })
                
                file_records = 0
                unknown_in_file = []
                
                for idx, row in df_extracted.iterrows():
                    sr_no = row['Sr.No']
                    particular = row['Particulars']
                    cwi_amount = row['As Per CWI']
                    rate = row['Rate per sq.feet']
                    
                    if pd.isna(particular) or str(particular).strip() == '':
                        continue
                    
                    standard_name = self.standardize_group_name(particular)
                    
                    if standard_name is None:
                        original_particular = str(particular).strip()
                        unknown_in_file.append(original_particular)
                        self.unknown_particulars.add(original_particular)
                        continue
                    
                    extracted_cwi_amount = self.extract_amount(cwi_amount)
                    extracted_rate = self.extract_amount(rate)
                    
                    record = {
                        'file': file_name,
                        'sr_no': sr_no,
                        'original_particular': str(particular),
                        'standard_group': standard_name,
                        'cwi_amount': extracted_cwi_amount,
                        'rate': extracted_rate,
                        'sqfeet_range': sqfeet_range
                    }
                    training_data.append(record)
                    file_records += 1
                
                print(f"  ✓ Loaded {file_records} records from {file_name}")
                
                if unknown_in_file:
                    print(f"  ⚠ Found {len(unknown_in_file)} unknown particulars in this file:")
                    for unknown in unknown_in_file[:5]:
                        print(f"    - {unknown}")
                    if len(unknown_in_file) > 5:
                        print(f"    ... and {len(unknown_in_file) - 5} more")
                
            except Exception as e:
                print(f"  ✗ Error processing {file_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*70}")
        print(f"Total training records loaded: {len(training_data)}")
        
        range_counts = {}
        for record in training_data:
            range_key = record['sqfeet_range']
            range_counts[range_key] = range_counts.get(range_key, 0) + 1
        
        print("\nDistribution by Sq Feet Range:")
        for range_key in sorted(range_counts.keys()):
            count = range_counts[range_key]
            print(f"  {range_key}: {count} records")
        
        group_counts = {}
        for record in training_data:
            group_key = record['standard_group']
            group_counts[group_key] = group_counts.get(group_key, 0) + 1
        
        print("\nDistribution by Grouped Category:")
        for group_key in sorted(group_counts.keys()):
            count = group_counts[group_key]
            print(f"  {group_key}: {count} records")
        
        if self.unknown_particulars:
            print(f"\n⚠ UNKNOWN PARTICULARS FOUND ({len(self.unknown_particulars)} unique):")
            print("These particulars were not matched to any standard group:")
            for unknown in sorted(self.unknown_particulars)[:10]:
                print(f"  - {unknown}")
            if len(self.unknown_particulars) > 10:
                print(f"  ... and {len(self.unknown_particulars) - 10} more")
            print("\nPlease review these and update the known_groups dictionary if needed.")
        
        return training_data
   
    def calculate_grouped_ranges(self, training_data):
        """Calculate CWI amount and rate ranges with WEIGHTED AVERAGES for each grouped category"""
        print("\n" + "="*70)
        print("Calculating ranges by Grouped Categories (WITH WEIGHTED AVERAGES)...")
        print("="*70)
        
        for record in training_data:
            sqfeet_range = record['sqfeet_range']
            standard_group = record['standard_group']
            cwi_amount = record['cwi_amount']
            rate = record['rate']
            
            if standard_group not in self.sqfeet_ranges[sqfeet_range]['particulars_data']:
                self.sqfeet_ranges[sqfeet_range]['particulars_data'][standard_group] = {
                    'cwi_amounts': [],
                    'rates': []
                }
            
            self.sqfeet_ranges[sqfeet_range]['particulars_data'][standard_group]['cwi_amounts'].append(cwi_amount)
            self.sqfeet_ranges[sqfeet_range]['particulars_data'][standard_group]['rates'].append(rate)
        
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            if not range_data['particulars_data']:
                continue
            
            print(f"\n{'─'*70}")
            print(f"Processing range: {sqfeet_range}")
            print(f"{'─'*70}")
            groups_in_range = 0
            
            for group_name, data in range_data['particulars_data'].items():
                cwi_amounts = np.array(data['cwi_amounts'])
                rates = np.array(data['rates'])
                
                cwi_stats = self._calculate_statistics(cwi_amounts, 'CWI Amount')
                rate_stats = self._calculate_statistics(rates, 'Rate')
                
                combined_stats = {
                    'cwi_amount_stats': cwi_stats,
                    'rate_stats': rate_stats
                }
                
                range_data['grouped_ranges'][group_name] = combined_stats
                groups_in_range += 1
                
                print(f"\n  Group: {group_name}")
                print(f"    CWI Threshold: ₹{cwi_stats['validation_range_lower']:.2f} - ₹{cwi_stats['validation_range_upper']:.2f}")
                print(f"    CWI Weighted Avg: ₹{cwi_stats['weighted_avg_min']:.2f} - ₹{cwi_stats['weighted_avg_max']:.2f}")
                print(f"    Rate Threshold: ₹{rate_stats['validation_range_lower']:.2f} - ₹{rate_stats['validation_range_upper']:.2f}")
                print(f"    Rate Weighted Avg: ₹{rate_stats['weighted_avg_min']:.2f} - ₹{rate_stats['weighted_avg_max']:.2f}")
                print(f"    Records: {cwi_stats['total_count']}")
                print(f"    Zero CWI: {cwi_stats['zero_count']} ({cwi_stats['zero_percentage']:.1f}%)")
                print(f"    Zero Rates: {rate_stats['zero_count']} ({rate_stats['zero_percentage']:.1f}%)")
            
            print(f"\n  ✓ Calculated ranges for {groups_in_range} grouped categories")
        
        print(f"\n{'='*70}")
        print(f"Summary by Sq Feet Range:")
        print(f"{'='*70}")
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            group_count = len(range_data['grouped_ranges'])
            if group_count > 0:
                print(f"  {sqfeet_range}: {group_count} grouped categories")
    
    def _calculate_statistics(self, values, value_type):
        """Calculate comprehensive statistics including WEIGHTED AVERAGES"""
        non_zero_values = values[values > 0]
        zero_values = values[values == 0]

        stats = {
            'total_count': len(values),
            'zero_count': len(zero_values),
            'non_zero_count': len(non_zero_values),
            'zero_percentage': (len(zero_values) / len(values)) * 100 if len(values) > 0 else 0,
            'zero_allowed': len(zero_values) > 0,
            'overall_min': float(np.min(values)),
            'overall_max': float(np.max(values)),
            'overall_mean': float(np.mean(values)),
            'overall_median': float(np.median(values)),
            'overall_std': float(np.std(values))
        }

        if len(non_zero_values) > 0:
            non_zero_stats = {
                'non_zero_min': float(np.min(non_zero_values)),
                'non_zero_max': float(np.max(non_zero_values)),
                'non_zero_mean': float(np.mean(non_zero_values)),
                'non_zero_median': float(np.median(non_zero_values)),
                'non_zero_std': float(np.std(non_zero_values))
            }
            stats.update(non_zero_stats)

        # === CALCULATE THRESHOLD RANGE (mean ± 2*std) ===
        if stats['zero_allowed']:
            threshold_lower = 0.0
        else:
            if stats['overall_std'] > 0:
                stat_lower = stats['overall_mean'] - 2 * stats['overall_std']
                threshold_lower = max(stats['overall_min'], stat_lower)
                threshold_lower = max(threshold_lower, 0)
            else:
                threshold_lower = stats['overall_min']

        if stats['overall_std'] > 0:
            stat_upper = stats['overall_mean'] + 2 * stats['overall_std']
            threshold_upper = max(stat_upper, stats['overall_max'])
        else:
            threshold_upper = stats['overall_max']

        threshold_lower = round(threshold_lower, 2)
        threshold_upper = round(threshold_upper, 2)

        # === CALCULATE WEIGHTED AVERAGE RANGE (mean ± 1*std) ===
        if stats['overall_std'] > 0:
            weighted_avg_min = stats['overall_mean'] - stats['overall_std']
            weighted_avg_max = stats['overall_mean'] + stats['overall_std']
        else:
            weighted_avg_min = stats['overall_mean']
            weighted_avg_max = stats['overall_mean']

        # ✅ CRITICAL: Ensure weighted range stays within threshold range
        weighted_avg_min = max(weighted_avg_min, threshold_lower)
        weighted_avg_max = min(weighted_avg_max, threshold_upper)

        weighted_avg_min = round(weighted_avg_min, 2)
        weighted_avg_max = round(weighted_avg_max, 2)

        stats.update({
            'validation_range_lower': threshold_lower,
            'validation_range_upper': threshold_upper,
            'weighted_avg_min': weighted_avg_min,
            'weighted_avg_max': weighted_avg_max
        })

        return stats
   
    def train_text_similarity(self):
        """Train text similarity model for group names"""
        print("\n" + "="*70)
        print("Training text similarity models for grouped categories...")
        print("="*70)
        
        group_names = list(self.known_groups.keys())
        
        if group_names:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            group_vectors = vectorizer.fit_transform(group_names)
            
            for sqfeet_range in self.sqfeet_ranges.keys():
                self.vectorizers[sqfeet_range] = vectorizer
                self.particular_vectors[sqfeet_range] = group_vectors
                self.particular_names[sqfeet_range] = group_names
            
            print(f"  ✓ Trained with {len(group_names)} standard group names")
            for group_name in group_names:
                print(f"    - {group_name}")
        else:
            print(f"  ⚠ No group names found for training")

    def find_similar_group(self, query_particular, sqfeet_range, threshold=0.5):
        """Find most similar group from standard groups"""
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
        """Validate CWI amount and rate using WEIGHTED AVERAGE ranges"""
        if sqfeet_range not in self.sqfeet_ranges:
            return {
                'cwi_valid': False,
                'rate_valid': False,
                'message': f"Invalid sq feet range: {sqfeet_range}"
            }
        
        range_data = self.sqfeet_ranges[sqfeet_range]
        
        standard_group = self.standardize_group_name(group_name)
        
        if standard_group is None:
            similar_group, similarity = self.find_similar_group(group_name, sqfeet_range)
            if similar_group:
                standard_group = similar_group
            else:
                return {
                    'cwi_valid': False,
                    'rate_valid': False,
                    'message': f"Unknown group: '{group_name}' - not found in standard groups"
                }
        
        if standard_group not in range_data['grouped_ranges']:
            return {
                'cwi_valid': False,
                'rate_valid': False,
                'message': f"Group '{standard_group}' not found in training data for {sqfeet_range} range"
            }
        
        group_stats = range_data['grouped_ranges'][standard_group]
        cwi_stats = group_stats['cwi_amount_stats']
        rate_stats = group_stats['rate_stats']
        
        # === VALIDATE CWI AMOUNT using WEIGHTED AVERAGES ===
        cwi_valid = False
        cwi_message = ""
        
        cwi_weighted_min = cwi_stats.get('weighted_avg_min', cwi_stats['validation_range_lower'])
        cwi_weighted_max = cwi_stats.get('weighted_avg_max', cwi_stats['validation_range_upper'])
        
        if cwi_amount == 0 and cwi_stats['zero_allowed']:
            cwi_valid = True
            cwi_message = f"Zero CWI amount allowed ({cwi_stats['zero_count']}/{cwi_stats['total_count']} cases)"
        elif cwi_weighted_min <= cwi_amount <= cwi_weighted_max:
            cwi_valid = True
            cwi_message = f"Within range: ₹{cwi_weighted_min:.2f} - ₹{cwi_weighted_max:.2f}"
        else:
            cwi_message = f"Outside range: ₹{cwi_weighted_min:.2f} - ₹{cwi_weighted_max:.2f}"
        
        # === VALIDATE RATE using WEIGHTED AVERAGES ===
        rate_valid = False
        rate_message = ""
        
        rate_weighted_min = rate_stats.get('weighted_avg_min', rate_stats['validation_range_lower'])
        rate_weighted_max = rate_stats.get('weighted_avg_max', rate_stats['validation_range_upper'])
        
        if rate == 0 and rate_stats['zero_allowed']:
            rate_valid = True
            rate_message = f"Zero rate allowed ({rate_stats['zero_count']}/{rate_stats['total_count']} cases)"
        elif rate_weighted_min <= rate <= rate_weighted_max:
            rate_valid = True
            rate_message = f"Within range: ₹{rate_weighted_min:.2f} - ₹{rate_weighted_max:.2f}"
        else:
            rate_message = f"Outside range: ₹{rate_weighted_min:.2f} - ₹{rate_weighted_max:.2f}"
        
        return {
            'cwi_valid': cwi_valid,
            'rate_valid': rate_valid,
            'standard_group': standard_group,
            'cwi_message': cwi_message,
            'rate_message': rate_message,
            'cwi_stats': cwi_stats,
            'rate_stats': rate_stats
        }
   
    def save_model(self, model_path):
        """Save trained model to file"""
        model_data = {
            'sqfeet_ranges': self.sqfeet_ranges,
            'vectorizers': self.vectorizers,
            'particular_vectors': self.particular_vectors,
            'particular_names': self.particular_names,
            'training_stats': self.training_stats,
            'known_groups': self.known_groups,
            'all_groups': self.all_groups,
            'new_groups_found': self.new_groups_found,
            'unknown_particulars': self.unknown_particulars
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to: {model_path}")

    def load_model(self, model_path):
        """Load trained model from file"""
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

        print(f"✓ Model loaded from: {model_path}")

    def display_training_summary(self):
        """Display comprehensive training summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TRAINING SUMMARY (GROUPED CATEGORIES - WITH WEIGHTED AVERAGES)")
        print("="*80)

        total_groups = 0
        total_cwi_records = 0
        total_cwi_zero_records = 0
        total_rate_records = 0
        total_rate_zero_records = 0

        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            range_groups = len(range_data['grouped_ranges'])

            if range_groups > 0:
                range_cwi_records = sum(stats['cwi_amount_stats']['total_count'] 
                                       for stats in range_data['grouped_ranges'].values())
                range_cwi_zeros = sum(stats['cwi_amount_stats']['zero_count'] 
                                     for stats in range_data['grouped_ranges'].values())
                range_rate_records = sum(stats['rate_stats']['total_count'] 
                                        for stats in range_data['grouped_ranges'].values())
                range_rate_zeros = sum(stats['rate_stats']['zero_count'] 
                                      for stats in range_data['grouped_ranges'].values())

                print(f"\nSq Feet Range: {sqfeet_range}")
                print(f"{'─'*70}")
                print(f"  Grouped categories: {range_groups}")
                print(f"  Total records: {range_cwi_records}")
                print(f"  CWI zero records: {range_cwi_zeros} ({(range_cwi_zeros/range_cwi_records)*100:.1f}%)")
                print(f"  Rate zero records: {range_rate_zeros} ({(range_rate_zeros/range_rate_records)*100:.1f}%)")

                print(f"\n  Groups in this range:")
                for group_name, stats in range_data['grouped_ranges'].items():
                    cwi_stats = stats['cwi_amount_stats']
                    rate_stats = stats['rate_stats']

                    group_marker = "🆕" if group_name in self.new_groups_found else "  "

                    print(f"    {group_marker} {group_name}")
                    print(f"      CWI Threshold: ₹{cwi_stats['validation_range_lower']:.2f} - "
                          f"₹{cwi_stats['validation_range_upper']:.2f}")
                    print(f"      CWI Weighted: ₹{cwi_stats['weighted_avg_min']:.2f} - "
                          f"₹{cwi_stats['weighted_avg_max']:.2f} "
                          f"(Zeros: {cwi_stats['zero_percentage']:.1f}%)")
                    print(f"      Rate Threshold: ₹{rate_stats['validation_range_lower']:.2f} - "
                          f"₹{rate_stats['validation_range_upper']:.2f}")
                    print(f"      Rate Weighted: ₹{rate_stats['weighted_avg_min']:.2f} - "
                          f"₹{rate_stats['weighted_avg_max']:.2f} "
                          f"(Zeros: {rate_stats['zero_percentage']:.1f}%)")

                total_groups += range_groups
                total_cwi_records += range_cwi_records
                total_cwi_zero_records += range_cwi_zeros
                total_rate_records += range_rate_records
                total_rate_zero_records += range_rate_zeros

        print(f"\n{'='*80}")
        print(f"OVERALL TOTALS:")
        print(f"{'='*80}")
        print(f"Total grouped categories: {total_groups}")
        print(f"Total CWI records: {total_cwi_records}")
        print(f"Total CWI zero records: {total_cwi_zero_records} ({(total_cwi_zero_records/total_cwi_records)*100:.1f}%)")
        print(f"Total Rate records: {total_rate_records}")
        print(f"Total Rate zero records: {total_rate_zero_records} ({(total_rate_zero_records/total_rate_records)*100:.1f}%)")

        print(f"\n{'='*80}")
        print(f"GROUPS BREAKDOWN:")
        print(f"{'='*80}")
        known_count = len([g for g in self.all_groups if g in self.known_groups.keys()])
        new_count = len(self.new_groups_found)
        print(f"Known/Standard groups: {known_count}")
        print(f"Newly discovered groups: {new_count}")
        print(f"Total unique groups: {len(self.all_groups)}")

        if self.new_groups_found:
            print(f"\n{'='*80}")
            print(f"🆕 NEW GROUPS DISCOVERED ({len(self.new_groups_found)}):")
            print(f"{'='*80}")
            print("These new groups were automatically added during training:")
            for new_group in sorted(self.new_groups_found):
                count = sum(1 for range_data in self.sqfeet_ranges.values() 
                           if new_group in range_data['grouped_ranges'])

                total_records = sum(range_data['grouped_ranges'][new_group]['cwi_amount_stats']['total_count']
                                  for range_data in self.sqfeet_ranges.values()
                                  if new_group in range_data['grouped_ranges'])

                print(f"  ✓ {new_group}")
                print(f"    - Found in {count} sq feet range(s)")
                print(f"    - Total records: {total_records}")
            print("\nThese groups will be included in validation.")
            print("Consider adding them to 'known_groups' for better standardization.")

        known_groups_used = [g for g in self.all_groups if g in self.known_groups.keys()]
        if known_groups_used:
            print(f"\n{'='*80}")
            print(f"KNOWN GROUPS USED ({len(known_groups_used)}):")
            print(f"{'='*80}")
            for known_group in sorted(known_groups_used):
                count = sum(1 for range_data in self.sqfeet_ranges.values() 
                           if known_group in range_data['grouped_ranges'])
                total_records = sum(range_data['grouped_ranges'][known_group]['cwi_amount_stats']['total_count']
                                  for range_data in self.sqfeet_ranges.values()
                                  if known_group in range_data['grouped_ranges'])
                print(f"  ✓ {known_group}")
                print(f"    - Found in {count} sq feet range(s)")
                print(f"    - Total records: {total_records}")

        if self.unknown_particulars:
            print(f"\n{'='*80}")
            print(f"⚠ UNKNOWN PARTICULARS ({len(self.unknown_particulars)}):")
            print(f"{'='*80}")
            print("These particulars were not matched and could not be auto-categorized.")
            print("Please review and update known_groups dictionary if needed.")
            for unknown in sorted(self.unknown_particulars)[:15]:
                print(f"  - {unknown}")
            if len(self.unknown_particulars) > 15:
                print(f"  ... and {len(self.unknown_particulars) - 15} more")

        print(f"\n{'='*80}")
        print("✅ Model is ready with WEIGHTED AVERAGES for validation!")
        print("✅ Validation uses WEIGHTED AVG ranges (mean ± 1*std)")
        print("✅ Threshold ranges (mean ± 2*std) stored for reference")
        print("="*80)

def main():
    print("="*80)
    print("CWI GROUPED VALIDATOR - HDFC Training Phase (With Weighted Averages)")
    print("="*80)
    
    validator = CWIGroupedValidator()
    
    training_folder = r"C:\Users\Tanush.Bidkar\Downloads\CWI validation HDFC\training files"
    
    training_data = validator.load_training_data(training_folder)
    
    if not training_data:
        print("\n" + "="*80)
        print("ERROR: No training data loaded!")
        print("="*80)
        print("\nPlease check:")
        print("1. File path is correct")
        print("2. Excel files exist in the folder")
        print("3. Files have 'Summary Sheet' tab")
        print("4. Filenames contain sq feet ranges (e.g., '500-750', '1001-1250')")
        print("5. 'Summary Sheet' has columns with 'CWI' keyword")
        return
    
    validator.calculate_grouped_ranges(training_data)
    validator.train_text_similarity()
    
    output_folder = r"C:\Users\Tanush.Bidkar\Downloads\CWI validation HDFC"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    model_path = os.path.join(output_folder, "trained_model_sqfeet_zero_aware_logic_grouped_HDFC.pkl")
    validator.save_model(model_path)
    
    validator.display_training_summary()
    
    report_path = os.path.join(output_folder, "training_report_sqfeet_zero_aware_grouped_weighted_avg_HDFC.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CWI GROUPED VALIDATOR - HDFC Training Report (With Weighted Averages)\n")
        f.write("="*80 + "\n\n")
        
        f.write("VALIDATION LOGIC:\n")
        f.write("-"*80 + "\n")
        f.write("- Threshold Range: mean ± 2*std (stored for reference)\n")
        f.write("- Weighted Avg Range: mean ± 1*std (USED FOR VALIDATION)\n")
        f.write("- Zero values handled separately based on training data\n\n")
        
        f.write("STANDARD GROUPS CONFIGURED:\n")
        f.write("-"*80 + "\n")
        for group_name, keywords in validator.known_groups.items():
            f.write(f"{group_name}: {', '.join(keywords)}\n")
        f.write("\n")
        
        for sqfeet_range, range_data in validator.sqfeet_ranges.items():
            if len(range_data['grouped_ranges']) > 0:
                f.write(f"\nSq Feet Range: {sqfeet_range}\n")
                f.write("-" * 80 + "\n")
                
                for group_name, stats in range_data['grouped_ranges'].items():
                    cwi_stats = stats['cwi_amount_stats']
                    rate_stats = stats['rate_stats']
                    
                    f.write(f"\nGroup: {group_name}\n")
                    f.write(f"{'─'*60}\n")
                    
                    f.write(f"CWI AMOUNT STATISTICS:\n")
                    f.write(f"  Total Count: {cwi_stats['total_count']}\n")
                    f.write(f"  Zero Count: {cwi_stats['zero_count']} ({cwi_stats['zero_percentage']:.1f}%)\n")
                    f.write(f"  Non-Zero Count: {cwi_stats['non_zero_count']}\n")
                    f.write(f"  Zero Allowed: {cwi_stats['zero_allowed']}\n")
                    f.write(f"  Overall Mean: ₹{cwi_stats['overall_mean']:.2f}\n")
                    f.write(f"  Overall Std: ₹{cwi_stats['overall_std']:.2f}\n")
                    if 'non_zero_mean' in cwi_stats:
                        f.write(f"  Non-Zero Mean: ₹{cwi_stats['non_zero_mean']:.2f}\n")
                        f.write(f"  Non-Zero Std: ₹{cwi_stats['non_zero_std']:.2f}\n")
                    f.write(f"  Threshold Range: ₹{cwi_stats['validation_range_lower']:.2f} - "
                           f"₹{cwi_stats['validation_range_upper']:.2f}\n")
                    f.write(f"  Weighted Avg Range (USED): ₹{cwi_stats['weighted_avg_min']:.2f} - "
                           f"₹{cwi_stats['weighted_avg_max']:.2f}\n")
                    
                    f.write(f"\nRATE STATISTICS:\n")
                    f.write(f"  Total Count: {rate_stats['total_count']}\n")
                    f.write(f"  Zero Count: {rate_stats['zero_count']} ({rate_stats['zero_percentage']:.1f}%)\n")
                    f.write(f"  Non-Zero Count: {rate_stats['non_zero_count']}\n")
                    f.write(f"  Zero Allowed: {rate_stats['zero_allowed']}\n")
                    f.write(f"  Overall Mean: ₹{rate_stats['overall_mean']:.2f}\n")
                    f.write(f"  Overall Std: ₹{rate_stats['overall_std']:.2f}\n")
                    if 'non_zero_mean' in rate_stats:
                        f.write(f"  Non-Zero Mean: ₹{rate_stats['non_zero_mean']:.2f}\n")
                        f.write(f"  Non-Zero Std: ₹{rate_stats['non_zero_std']:.2f}\n")
                    f.write(f"  Threshold Range: ₹{rate_stats['validation_range_lower']:.2f} - "
                           f"₹{rate_stats['validation_range_upper']:.2f}\n")
                    f.write(f"  Weighted Avg Range (USED): ₹{rate_stats['weighted_avg_min']:.2f} - "
                           f"₹{rate_stats['weighted_avg_max']:.2f}\n")
                    f.write("\n")
                
                f.write("\n")
        
        if validator.unknown_particulars:
            f.write(f"\n{'='*80}\n")
            f.write(f"UNKNOWN PARTICULARS ({len(validator.unknown_particulars)}):\n")
            f.write(f"{'='*80}\n")
            f.write("These particulars were not matched to any standard group.\n\n")
            for unknown in sorted(validator.unknown_particulars):
                f.write(f"  - {unknown}\n")
    
    print(f"\n✓ Enhanced training report saved to: {report_path}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()