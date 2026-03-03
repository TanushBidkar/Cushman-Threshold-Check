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

class CWIRateValidator:
    def __init__(self):
        # Store data segregated by sq feet ranges (250 sqft intervals)
        self.sqfeet_ranges = {
            '500-750': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '751-1000': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '1001-1250': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '1251-1500': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '1501-1750': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '1751-2000': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '2001-2250': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '2251-2500': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '2501-2750': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '2751-3000': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '3001-3250': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '3251-3500': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '3501-3750': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '3751-4000': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '4001-4250': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '4251-4500': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '4501-4750': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '4751-5000': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '5000+': {'rate_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)}
        }
        # Text similarity models for each range
        self.vectorizers = {}
        self.particular_vectors = {}
        self.particular_names = {}
        self.training_stats = {}
   
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
   
    def extract_rate(self, rate_value):
        """Extract numeric rate from various formats"""
        if pd.isna(rate_value):
            return 0
       
        rate_str = str(rate_value).strip()
        # Remove currency symbols and commas
        rate_str = rate_str.replace('₹', '').replace('Rs', '').replace(',', '').replace('INR', '').strip()
        # Handle negative values
        rate_str = rate_str.replace('(', '-').replace(')', '')
        numbers = re.findall(r'-?\d+\.?\d*', rate_str)
        if numbers:
            return float(numbers[0])
        return 0

    def load_training_data(self, folder_path):
        """Load all Excel files from training folder and segregate by sq feet extracted from filename"""
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
            print(f"Processing: {file_name}")
           
            # Extract sq feet range from filename
            sqfeet_range = self.extract_sqfeet_from_filename(file_name)
            if sqfeet_range is None:
                print(f"Warning: Could not extract sq feet range from filename: {file_name}")
                print("Please ensure filename contains sq feet range (e.g., '500-750', '1001-1250', etc.)")
                continue
           
            print(f"  Detected sq feet range: {sqfeet_range}")
           
            try:
                df = pd.read_excel(file_path, sheet_name='Extracted Data')
               
                # Expected columns (Particulars and Rate)
                expected_cols = ['Sr.No', 'Particulars', 'Rate']
                df_cols = df.columns.tolist()
                col_mapping = {}
               
                for expected_col in expected_cols:
                    found = False
                    for df_col in df_cols:
                        if expected_col.lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '') in str(df_col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', ''):
                            col_mapping[expected_col] = df_col
                            found = True
                            break
                    if not found and expected_col != 'Sr.No':  # Sr.No is optional
                        print(f"Warning: Column '{expected_col}' not found in {file_name}")
               
                if len(col_mapping) < 2:  # At least Particulars and Rate
                    print(f"Skipping {file_name} - missing required columns")
                    print(f"Found columns: {list(col_mapping.keys())}")
                    print(f"Available columns in file: {df_cols}")
                    continue
               
                df_renamed = df.rename(columns={v: k for k, v in col_mapping.items()})
               
                file_records = 0
                for idx, row in df_renamed.iterrows():
                    sr_no = row.get('Sr.No', '')
                    particular = row.get('Particulars', '')
                    rate = row.get('Rate', 0)
                   
                    if pd.isna(particular) or str(particular).strip() == '':
                        continue
                   
                    cleaned_particular = self.clean_text(particular)
                    extracted_rate = self.extract_rate(rate)
                   
                    if cleaned_particular:
                        record = {
                            'file': file_name,
                            'sr_no': sr_no,
                            'particular': cleaned_particular,
                            'original_particular': str(particular),
                            'rate': extracted_rate,
                            'sqfeet_range': sqfeet_range
                        }
                        training_data.append(record)
                        file_records += 1
               
                print(f"  ✓ Loaded {file_records} records from {file_name}")
               
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue
       
        print(f"\nTotal training records loaded: {len(training_data)}")
       
        # Display distribution by sq feet range
        range_counts = {}
        for record in training_data:
            range_key = record['sqfeet_range']
            range_counts[range_key] = range_counts.get(range_key, 0) + 1
       
        print("\nDistribution by Sq Feet Range (from filenames):")
        for range_key, count in sorted(range_counts.items()):
            print(f"  {range_key}: {count} records")
       
        return training_data
   
    def calculate_rate_ranges(self, training_data):
        """Calculate rate ranges for each type of particular within each sq feet range"""
        print("\nCalculating rate ranges by Sq Feet categories (EXCLUDING ZEROS FROM WEIGHTED AVG)...")
       
        # Group data by sq feet range and particular
        for record in training_data:
            sqfeet_range = record['sqfeet_range']
            particular = record['particular']
            rate = record['rate']
            file_name = record['file']
            original_particular = record['original_particular']
           
            self.sqfeet_ranges[sqfeet_range]['particulars_data'][particular].append(rate)
            # Store file-level data for tracking
            self.sqfeet_ranges[sqfeet_range]['particulars_file_data'][particular].append({
                'file': file_name,
                'rate': rate,
                'original_particular': original_particular
            })
       
        # Calculate comprehensive statistics for each range and particular combination
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            if not range_data['particulars_data']:  # Skip empty ranges
                continue
               
            print(f"\nProcessing range: {sqfeet_range}")
            particulars_in_range = 0
           
            for particular, rates in range_data['particulars_data'].items():
                rates = np.array(rates)
                non_zero_rate = rates[rates > 0]
                zero_rate = rates[rates == 0]
               
                # Overall statistics (including zeros)
                overall_stats = {
                    'total_count': len(rates),
                    'zero_count': len(zero_rate),
                    'non_zero_count': len(non_zero_rate),
                    'zero_percentage': (len(zero_rate) / len(rates)) * 100,
                    'zero_allowed': len(zero_rate) > 0,
                    'overall_min': float(np.min(rates)),
                    'overall_max': float(np.max(rates)),
                    'overall_mean': float(np.mean(rates)),
                    'overall_median': float(np.median(rates)),
                    'overall_std': float(np.std(rates))
                }
               
                # Non-zero statistics (for reference)
                if len(non_zero_rate) > 0:
                    non_zero_stats = {
                        'non_zero_min': float(np.min(non_zero_rate)),
                        'non_zero_max': float(np.max(non_zero_rate)),
                        'non_zero_mean': float(np.mean(non_zero_rate)),
                        'non_zero_median': float(np.median(non_zero_rate)),
                        'non_zero_std': float(np.std(non_zero_rate))
                    }
                    overall_stats.update(non_zero_stats)
               
                # Calculate THRESHOLD RANGE (for validation) - mean ± 2*std
                if overall_stats['overall_std'] > 0:
                    threshold_min = max(0, overall_stats['overall_mean'] - 2 * overall_stats['overall_std'])
                    threshold_max = overall_stats['overall_mean'] + 2 * overall_stats['overall_std']
                   
                    # If training data contains zeros, ensure lower bound is 0
                    if overall_stats['zero_allowed']:
                        threshold_min = 0
                   
                    # Ensure bounds cover all training data
                    threshold_min = min(threshold_min, overall_stats['overall_min'])
                    threshold_max = max(threshold_max, overall_stats['overall_max'])
                else:
                    # If std is 0, all values are the same
                    if overall_stats['zero_allowed']:
                        threshold_min = 0
                        threshold_max = overall_stats['overall_max'] * 1.2 if overall_stats['overall_max'] > 0 else 0
                    else:
                        threshold_min = overall_stats['overall_min'] * 0.8
                        threshold_max = overall_stats['overall_max'] * 1.2
               
                # ✅ Calculate WEIGHTED AVERAGE RANGE (EXCLUDING ZEROS) - mean ± 1*std
                if len(non_zero_rate) > 0:
                    non_zero_mean = float(np.mean(non_zero_rate))
                    non_zero_std = float(np.std(non_zero_rate))
                   
                    if non_zero_std > 0:
                        weighted_avg_min = max(0, non_zero_mean - non_zero_std)
                        weighted_avg_max = non_zero_mean + non_zero_std
                    else:
                        # If std is 0, all non-zero values are the same
                        weighted_avg_min = non_zero_mean
                        weighted_avg_max = non_zero_mean
                    
                    # ✅ Ensure weighted range stays within threshold range
                    weighted_avg_min = max(weighted_avg_min, threshold_min)
                    weighted_avg_max = min(weighted_avg_max, threshold_max)
                else:
                    # If all values are zero
                    weighted_avg_min = 0
                    weighted_avg_max = 0
               
                # Round rate thresholds to 2 decimal places
                threshold_min = round(threshold_min, 2)
                threshold_max = round(threshold_max, 2)
                weighted_avg_min = round(weighted_avg_min, 2)
                weighted_avg_max = round(weighted_avg_max, 2)
                
                # ✅ Store file information for ACTUAL threshold min/max
                file_data = range_data['particulars_file_data'][particular]
                
                min_files = []
                max_files = []
                
                # Use overall_min and overall_max (the ACTUAL min/max from training data)
                actual_min = overall_stats['overall_min']
                actual_max = overall_stats['overall_max']
                
                for item in file_data:
                    item_rate = item['rate']
                    
                    # Check if rate matches ACTUAL min (with small tolerance for floats)
                    if abs(item_rate - actual_min) < 0.01:
                        min_files.append(item)
                    
                    # Check if rate matches ACTUAL max (with small tolerance for floats)
                    if abs(item_rate - actual_max) < 0.01:
                        max_files.append(item)
                
                overall_stats.update({
                    'validation_range_lower': threshold_min,
                    'validation_range_upper': threshold_max,
                    'weighted_avg_min': weighted_avg_min,
                    'weighted_avg_max': weighted_avg_max,
                    'min_threshold_files': min_files,
                    'max_threshold_files': max_files
                })
               
                range_data['rate_ranges'][particular] = overall_stats
                particulars_in_range += 1
           
            print(f"  ✓ Calculated ranges for {particulars_in_range} unique particulars")
       
        # Display summary
        print(f"\nSummary by Sq Feet Range:")
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            particular_count = len(range_data['rate_ranges'])
            if particular_count > 0:
                print(f"  {sqfeet_range}: {particular_count} unique particulars")
           
    def train_text_similarity(self):
        """Train text similarity model for each sq feet range"""
        print("\nTraining text similarity models for each Sq Feet range...")
       
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            particular_names = list(range_data['rate_ranges'].keys())
           
            if particular_names:
                vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                particular_vectors = vectorizer.fit_transform(particular_names)
               
                self.vectorizers[sqfeet_range] = vectorizer
                self.particular_vectors[sqfeet_range] = particular_vectors
                self.particular_names[sqfeet_range] = particular_names
               
                print(f"  ✓ {sqfeet_range}: Trained with {len(particular_names)} particulars")
            else:
                print(f"  ⚠ {sqfeet_range}: No particulars found for training")

    def find_similar_particular(self, query_particular, sqfeet_range, threshold=0.5):
        """Find most similar particular from training data within specific sq feet range"""
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
   
    def validate_rate(self, particular, rate, sqfeet_range):
        """✅ Validate rate using WEIGHTED AVERAGE ranges (excluding zeros)"""
        if sqfeet_range not in self.sqfeet_ranges:
            return False, f"Invalid sq feet range: {sqfeet_range}", None, None, None, None, None, None
       
        range_data = self.sqfeet_ranges[sqfeet_range]
       
        # First try exact match
        if particular in range_data['rate_ranges']:
            stats = range_data['rate_ranges'][particular]
           
            # ✅ Use weighted averages for validation
            validation_min = stats['weighted_avg_min']
            validation_max = stats['weighted_avg_max']
            
            # Keep threshold values for backward compatibility
            threshold_min = stats['validation_range_lower']
            threshold_max = stats['validation_range_upper']
            
            # Get min and max threshold file information
            min_threshold_files = stats.get('min_threshold_files', [])
            max_threshold_files = stats.get('max_threshold_files', [])
           
            # Special handling for zero rates
            if rate == 0 and stats['zero_allowed']:
                return True, f"Zero rate allowed (appears in {stats['zero_count']}/{stats['total_count']} training cases)", threshold_min, threshold_max, validation_min, validation_max, min_threshold_files, max_threshold_files
           
            # ✅ Validate against weighted average range
            is_valid = validation_min <= rate <= validation_max
            
            range_msg = f"Range: ₹{validation_min:.2f} - ₹{validation_max:.2f}"
            zero_info = f" (Zeros: {stats['zero_percentage']:.1f}%)" if stats['zero_allowed'] else ""
            return is_valid, f"Exact match - {range_msg}{zero_info}", threshold_min, threshold_max, validation_min, validation_max, min_threshold_files, max_threshold_files
       
        # Try similarity matching
        similar_particular, similarity = self.find_similar_particular(particular, sqfeet_range)
        if similar_particular:
            stats = range_data['rate_ranges'][similar_particular]
           
            # ✅ Use weighted averages for validation
            validation_min = stats['weighted_avg_min']
            validation_max = stats['weighted_avg_max']
            
            # Keep threshold values for backward compatibility
            threshold_min = stats['validation_range_lower']
            threshold_max = stats['validation_range_upper']
            
            # Get min and max threshold file information
            min_threshold_files = stats.get('min_threshold_files', [])
            max_threshold_files = stats.get('max_threshold_files', [])
           
            # Special handling for zero rates
            if rate == 0 and stats['zero_allowed']:
                return True, f"Zero rate allowed via similar match ({similarity:.2f}) - appears in {stats['zero_count']}/{stats['total_count']} training cases", threshold_min, threshold_max, validation_min, validation_max, min_threshold_files, max_threshold_files
           
            # ✅ Validate against weighted average range
            is_valid = validation_min <= rate <= validation_max
            
            range_msg = f"Range: ₹{validation_min:.2f} - ₹{validation_max:.2f}"
            zero_info = f" (Zeros: {stats['zero_percentage']:.1f}%)" if stats['zero_allowed'] else ""
            return is_valid, f"Similar match ({similarity:.2f}) - {range_msg}{zero_info}", threshold_min, threshold_max, validation_min, validation_max, min_threshold_files, max_threshold_files
       
        return False, "No matching particular found", None, None, None, None, None, None
   
    def save_model(self, model_path):
        """Save trained model to file"""
        model_data = {
            'sqfeet_ranges': self.sqfeet_ranges,
            'vectorizers': self.vectorizers,
            'particular_vectors': self.particular_vectors,
            'particular_names': self.particular_names,
            'training_stats': self.training_stats
        }
       
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
       
        print(f"✓ Model saved to: {model_path}")

    def load_model(self, model_path):
        """Load trained model from file"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
       
        self.sqfeet_ranges = model_data['sqfeet_ranges']
        self.vectorizers = model_data['vectorizers']
        self.particular_vectors = model_data['particular_vectors']
        self.particular_names = model_data['particular_names']
        self.training_stats = model_data['training_stats']
       
        print(f"✓ Model loaded from: {model_path}")

    def display_training_summary(self):
        """Display comprehensive training summary"""
        print("\n" + "="*70)
        print("COMPREHENSIVE TRAINING SUMMARY (WEIGHTED AVG EXCLUDES ZEROS - RATE)")
        print("="*70)
       
        total_particulars = 0
        total_records = 0
        total_zero_records = 0
       
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            range_particulars = len(range_data['rate_ranges'])
            range_records = sum(stats['total_count'] for stats in range_data['rate_ranges'].values())
            range_zeros = sum(stats['zero_count'] for stats in range_data['rate_ranges'].values())
           
            if range_particulars > 0:
                print(f"\nSq Feet Range: {sqfeet_range}")
                print(f"  Unique particulars: {range_particulars}")
                print(f"  Total records: {range_records}")
                print(f"  Zero rate records: {range_zeros} ({(range_zeros/range_records)*100:.1f}%)")
               
                zero_allowed_count = sum(1 for stats in range_data['rate_ranges'].values() if stats['zero_allowed'])
                print(f"  Particulars allowing zeros: {zero_allowed_count}/{range_particulars}")
               
                # Average weighted range (excluding zeros)
                weighted_ranges = [stats['weighted_avg_max'] - stats['weighted_avg_min']
                                   for stats in range_data['rate_ranges'].values() if 'weighted_avg_max' in stats]
               
                if weighted_ranges:
                    print(f"  Average weighted range size (excluding zeros): ₹{np.mean(weighted_ranges):.2f}")
               
                # Show top 3 particulars with their zero statistics
                sorted_particulars = sorted(range_data['rate_ranges'].items(),
                                            key=lambda x: x[1]['total_count'], reverse=True)
                print(f"  Top particulars:")
                for i, (particular, stats) in enumerate(sorted_particulars[:3]):
                    zero_pct = stats['zero_percentage']
                    print(f"    {i+1}. {particular[:35]}... (Count: {stats['total_count']}, Zeros: {zero_pct:.1f}%)")
               
                total_particulars += range_particulars
                total_records += range_records
                total_zero_records += range_zeros
       
        print(f"\nOVERALL TOTALS:")
        print(f"Total unique particulars: {total_particulars}")
        print(f"Total training records: {total_records}")
        print(f"Total zero rate records: {total_zero_records} ({(total_zero_records/total_records)*100:.1f}%)")
        print("\nModel is ready for validation with WEIGHTED AVG (EXCLUDING ZEROS) for RATES!")
        print("="*70)
def main():
    print("CWI Validator - Rate Training Phase (Weighted Avg Excludes Zeros, 250 sqft intervals)")
    print("="*80)
   
    validator = CWIRateValidator()
   
    # Specific path for ICICI
    training_folder = r"C:\Users\Tanush.Bidkar\Downloads\Axis Measurement Sheet\training files"
   
    training_data = validator.load_training_data(training_folder)
   
    if not training_data:
        print("No training data loaded. Please check your file path and data format.")
        print("\nMake sure your Excel filenames contain sq feet ranges like:")
        print("  - 'data_500-750.xlsx'")
        print("  - 'cwi_1001-1250.xlsx'")
        print("  - 'project_3001-3250.xlsx'")
        return
   
    # Calculate ranges specifically for RATES
    validator.calculate_rate_ranges(training_data)
    validator.train_text_similarity()
   
    output_folder = r"C:\Users\Tanush.Bidkar\Downloads\Axis Measurement Sheet"
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the RATE model
    model_path = os.path.join(output_folder, "trained_model_sqfeet_zero_aware_logic_rate_AXIS.pkl")
    validator.save_model(model_path)
   
    validator.display_training_summary()
   
    # Generate enhanced detailed report for RATES
    report_path = os.path.join(output_folder, "training_report_sqfeet_weighted_avg_rate.txt")
    
    print(f"\nGenerating detailed report at: {report_path}...")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CWI Validator Enhanced Training Report (Weighted Avg Excludes Zeros - RATE)\n")
            f.write("="*80 + "\n\n")
            
            for sqfeet_range, range_data in validator.sqfeet_ranges.items():
                # Check for rate_ranges
                if len(range_data['rate_ranges']) > 0:
                    f.write(f"Sq Feet Range: {sqfeet_range}\n")
                    f.write("-" * 40 + "\n")
                    
                    for particular, stats in range_data['rate_ranges'].items():
                        f.write(f"Particular: {particular}\n")
                        f.write(f"  Total Count: {stats['total_count']}\n")
                        f.write(f"  Zero Count: {stats['zero_count']} ({stats['zero_percentage']:.1f}%)\n")
                        f.write(f"  Non-Zero Count: {stats['non_zero_count']}\n")
                        f.write(f"  Zero Allowed: {stats['zero_allowed']}\n")
                        f.write(f"  Overall Mean: ₹{stats['overall_mean']:.2f}\n")
                        f.write(f"  Overall Std: ₹{stats['overall_std']:.2f}\n")
                        
                        if 'non_zero_mean' in stats:
                            f.write(f"  Non-Zero Mean: ₹{stats['non_zero_mean']:.2f}\n")
                            f.write(f"  Non-Zero Std: ₹{stats['non_zero_std']:.2f}\n")
                        
                        if 'validation_range_lower' in stats:
                            f.write(f"  Threshold Range: ₹{stats['validation_range_lower']:.2f} - ₹{stats['validation_range_upper']:.2f}\n")
                        
                        if 'weighted_avg_min' in stats:
                            f.write(f"  Weighted Avg Range (Excluding Zeros): ₹{stats['weighted_avg_min']:.2f} - ₹{stats['weighted_avg_max']:.2f}\n")
                        
                        # Write files information for Min Rate
                        if stats.get('min_threshold_files'):
                            f.write(f"  Files with Min Rate:\n")
                            for mf in stats['min_threshold_files'][:5]:  # Show first 5
                                f.write(f"    - {mf['file']}: {mf['original_particular']}\n")
                            if len(stats['min_threshold_files']) > 5:
                                f.write(f"    ... and {len(stats['min_threshold_files']) - 5} more\n")
                        
                        # Write files information for Max Rate
                        if stats.get('max_threshold_files'):
                            f.write(f"  Files with Max Rate:\n")
                            for mf in stats['max_threshold_files'][:5]:  # Show first 5
                                f.write(f"    - {mf['file']}: {mf['original_particular']}\n")
                            if len(stats['max_threshold_files']) > 5:
                                f.write(f"    ... and {len(stats['max_threshold_files']) - 5} more\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
       
        print(f"✓ Enhanced training report saved to: {report_path}")

    except Exception as e:
        print(f"Error saving report: {e}")

if __name__ == "__main__":
    main()