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

class CWIValidator:
    def __init__(self):
        # Store data segregated by sq feet ranges (250 sqft intervals)
        self.sqfeet_ranges = {
            '500-1000': {'amount_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '1001-1500': {'amount_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '1501-2000': {'amount_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '2001-2500': {'amount_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '2501-3000': {'amount_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '3001-3500': {'amount_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '3501-4000': {'amount_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '4001-4500': {'amount_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '4501-5000': {'amount_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)},
            '5000+': {'amount_ranges': {}, 'particulars_data': defaultdict(list), 'particulars_file_data': defaultdict(list)}
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

        return None
   
    def clean_text(self, text):
        """Clean and normalize text for better matching"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text
   
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
               
                # Expected columns (Particulars and Amount)
                expected_cols = ['Sr.No', 'Particulars', 'As Per CWI (Amount)']
                df_cols = df.columns.tolist()
                col_mapping = {}
               
                for expected_col in expected_cols:
                    found = False
                    for df_col in df_cols:
                        if expected_col.lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '') in str(df_col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', ''):
                            col_mapping[expected_col] = df_col
                            found = True
                            break
                    if not found:
                        print(f"Warning: Column '{expected_col}' not found in {file_name}")
               
                if len(col_mapping) < 2:  # At least Particulars and Amount
                    print(f"Skipping {file_name} - missing required columns")
                    print(f"Found columns: {list(col_mapping.keys())}")
                    print(f"Available columns in file: {df_cols}")
                    continue
               
                df_renamed = df.rename(columns={v: k for k, v in col_mapping.items()})
               
                file_records = 0
                for idx, row in df_renamed.iterrows():
                    sr_no = row.get('Sr.No', '')
                    particular = row.get('Particulars', '')
                    amount = row.get('As Per CWI (Amount)', 0)
                   
                    if pd.isna(particular) or str(particular).strip() == '':
                        continue
                   
                    cleaned_particular = self.clean_text(particular)
                    extracted_amount = self.extract_amount(amount)
                   
                    if cleaned_particular:
                        record = {
                            'file': file_name,
                            'sr_no': sr_no,
                            'particular': cleaned_particular,
                            'original_particular': str(particular),
                            'amount': extracted_amount,
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
   
    def calculate_amount_ranges(self, training_data):
        """✅ Calculate amount ranges EXCLUDING ZEROS for both threshold and weighted avg"""
        print("\nCalculating amount ranges by Sq Feet categories (EXCLUDING ZEROS FROM ALL CALCULATIONS)...")
       
        # Group data by sq feet range and particular
        for record in training_data:
            sqfeet_range = record['sqfeet_range']
            particular = record['particular']
            amount = record['amount']
            file_name = record['file']
            original_particular = record['original_particular']
           
            self.sqfeet_ranges[sqfeet_range]['particulars_data'][particular].append(amount)
            # Store file-level data for tracking
            self.sqfeet_ranges[sqfeet_range]['particulars_file_data'][particular].append({
                'file': file_name,
                'amount': amount,
                'original_particular': original_particular
            })
       
        # Calculate comprehensive statistics for each range and particular combination
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            if not range_data['particulars_data']:  # Skip empty ranges
                continue
               
            print(f"\nProcessing range: {sqfeet_range}")
            particulars_in_range = 0
           
            for particular, amounts in range_data['particulars_data'].items():
                amounts = np.array(amounts)
                
                # ✅ SEPARATE: Non-zero amounts for calculations
                non_zero_amt = amounts[amounts > 0]
                zero_amt = amounts[amounts == 0]
               
                # Overall statistics (including zeros) - FOR TRACKING ONLY
                overall_stats = {
                    'total_count': len(amounts),
                    'zero_count': len(zero_amt),
                    'non_zero_count': len(non_zero_amt),
                    'zero_percentage': (len(zero_amt) / len(amounts)) * 100,
                    'zero_allowed': len(zero_amt) > 0,
                }
                
                # ✅ SKIP particulars that have ONLY zeros
                if len(non_zero_amt) == 0:
                    print(f"  ⚠️  Skipping '{particular[:40]}...' - ALL values are zero")
                    continue
               
                # ✅ NON-ZERO statistics (used for ALL calculations)
                overall_stats.update({
                    'overall_min': float(np.min(non_zero_amt)),  # Min of NON-ZERO
                    'overall_max': float(np.max(non_zero_amt)),  # Max of NON-ZERO
                    'overall_mean': float(np.mean(non_zero_amt)),  # Mean of NON-ZERO
                    'overall_median': float(np.median(non_zero_amt)),  # Median of NON-ZERO
                    'overall_std': float(np.std(non_zero_amt))  # Std of NON-ZERO
                })
               
                # ✅ Calculate THRESHOLD RANGE using NON-ZERO data - mean ± 2*std
                non_zero_mean = overall_stats['overall_mean']
                non_zero_std = overall_stats['overall_std']
                
                if non_zero_std > 0:
                    threshold_min = non_zero_mean - 2 * non_zero_std  # Remove max(0, ...)
                    threshold_max = non_zero_mean + 2 * non_zero_std
                    
                    # ✅ CRITICAL: Ensure bounds cover all NON-ZERO training data (use max for min, not min!)
                    threshold_min = max(threshold_min, overall_stats['overall_min'])  # Can't be LESS than actual min
                    threshold_max = max(threshold_max, overall_stats['overall_max'])  # Can't be LESS than actual max
                    
                    # ✅ Ensure threshold_min is at least 0 (for currency)
                    threshold_min = max(threshold_min, 0)
                else:
                    # If std is 0, all non-zero values are the same
                    threshold_min = overall_stats['overall_min']
                    threshold_max = overall_stats['overall_max']
               
                # ✅ Calculate WEIGHTED AVERAGE RANGE using NON-ZERO data - mean ± 1*std
                if non_zero_std > 0:
                    weighted_avg_min = non_zero_mean - non_zero_std
                    weighted_avg_max = non_zero_mean + non_zero_std
                else:
                    # If std is 0, all non-zero values are the same
                    weighted_avg_min = non_zero_mean
                    weighted_avg_max = non_zero_mean

                # ✅ CRITICAL: Ensure weighted range stays within threshold range AND is never below threshold_min
                weighted_avg_min = max(weighted_avg_min, threshold_min)  # Can't be less than actual minimum
                weighted_avg_max = min(weighted_avg_max, threshold_max)  # Can't be more than actual maximum

                # ✅ Additional safety: Ensure weighted_avg_min is at least 0 (for currency/amounts)
                weighted_avg_min = max(weighted_avg_min, 0)
               
                # Round amount thresholds to 2 decimal places (for currency)
                threshold_min = round(threshold_min, 2)
                threshold_max = round(threshold_max, 2)
                weighted_avg_min = round(weighted_avg_min, 2)
                weighted_avg_max = round(weighted_avg_max, 2)
                
                # ✅ Store file information for ACTUAL NON-ZERO threshold min/max
                file_data = range_data['particulars_file_data'][particular]
                
                min_files = []
                max_files = []
                
                # Use overall_min and overall_max (the ACTUAL NON-ZERO min/max from training data)
                actual_min = overall_stats['overall_min']
                actual_max = overall_stats['overall_max']
                
                for item in file_data:
                    item_amt = item['amount']
                    
                    # ✅ SKIP zero amounts when tracking files
                    if item_amt == 0:
                        continue
                    
                    # Check if amount matches ACTUAL NON-ZERO min (with small tolerance for floats)
                    if abs(item_amt - actual_min) < 0.01:
                        min_files.append(item)
                    
                    # Check if amount matches ACTUAL NON-ZERO max (with small tolerance for floats)
                    if abs(item_amt - actual_max) < 0.01:
                        max_files.append(item)
                
                overall_stats.update({
                    'validation_range_lower': threshold_min,
                    'validation_range_upper': threshold_max,
                    'weighted_avg_min': weighted_avg_min,
                    'weighted_avg_max': weighted_avg_max,
                    'min_threshold_files': min_files,
                    'max_threshold_files': max_files
                })
               
                range_data['amount_ranges'][particular] = overall_stats
                particulars_in_range += 1
           
            print(f"  ✓ Calculated ranges for {particulars_in_range} unique particulars (zeros excluded)")
       
        # Display summary
        print(f"\nSummary by Sq Feet Range:")
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            particular_count = len(range_data['amount_ranges'])
            if particular_count > 0:
                print(f"  {sqfeet_range}: {particular_count} unique particulars")
           
    def train_text_similarity(self):
        """Train text similarity model for each sq feet range"""
        print("\nTraining text similarity models for each Sq Feet range...")
       
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            particular_names = list(range_data['amount_ranges'].keys())
           
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
   
    def validate_amount(self, particular, amount, sqfeet_range):
        """✅ Validate amount using WEIGHTED AVERAGE ranges (calculated from NON-ZERO values)"""
        if sqfeet_range not in self.sqfeet_ranges:
            return False, f"Invalid sq feet range: {sqfeet_range}", None, None, None, None, None, None
       
        range_data = self.sqfeet_ranges[sqfeet_range]
       
        # First try exact match
        if particular in range_data['amount_ranges']:
            stats = range_data['amount_ranges'][particular]
           
            # ✅ Use weighted averages for validation (calculated from non-zero values)
            validation_min = stats['weighted_avg_min']
            validation_max = stats['weighted_avg_max']
            
            # Keep threshold values for display
            threshold_min = stats['validation_range_lower']
            threshold_max = stats['validation_range_upper']
            
            # Get min and max threshold file information
            min_threshold_files = stats.get('min_threshold_files', [])
            max_threshold_files = stats.get('max_threshold_files', [])
           
            # ✅ Special handling for ZERO amounts in test file
            if amount == 0:
                if stats['zero_allowed']:
                    return True, f"Zero amount allowed (appears in {stats['zero_count']}/{stats['total_count']} training cases)", threshold_min, threshold_max, validation_min, validation_max, min_threshold_files, max_threshold_files
                else:
                    return False, f"Zero amount NOT allowed - Expected range: ₹{validation_min:.2f} - ₹{validation_max:.2f}", threshold_min, threshold_max, validation_min, validation_max, min_threshold_files, max_threshold_files
           
            # ✅ Validate against weighted average range (for NON-ZERO test amounts)
            is_valid = validation_min <= amount <= validation_max
            
            range_msg = f"Range: ₹{validation_min:.2f} - ₹{validation_max:.2f}"
            zero_info = f" (Zeros in training: {stats['zero_percentage']:.1f}%)" if stats['zero_allowed'] else ""
            return is_valid, f"Exact match - {range_msg}{zero_info}", threshold_min, threshold_max, validation_min, validation_max, min_threshold_files, max_threshold_files
       
        # Try similarity matching
        similar_particular, similarity = self.find_similar_particular(particular, sqfeet_range)
        if similar_particular:
            stats = range_data['amount_ranges'][similar_particular]
           
            # ✅ Use weighted averages for validation (calculated from non-zero values)
            validation_min = stats['weighted_avg_min']
            validation_max = stats['weighted_avg_max']
            
            # Keep threshold values for display
            threshold_min = stats['validation_range_lower']
            threshold_max = stats['validation_range_upper']
            
            # Get min and max threshold file information
            min_threshold_files = stats.get('min_threshold_files', [])
            max_threshold_files = stats.get('max_threshold_files', [])
           
            # ✅ Special handling for ZERO amounts in test file
            if amount == 0:
                if stats['zero_allowed']:
                    return True, f"Zero amount allowed via similar match ({similarity:.2f}) - appears in {stats['zero_count']}/{stats['total_count']} training cases", threshold_min, threshold_max, validation_min, validation_max, min_threshold_files, max_threshold_files
                else:
                    return False, f"Zero amount NOT allowed via similar match ({similarity:.2f}) - Expected range: ₹{validation_min:.2f} - ₹{validation_max:.2f}", threshold_min, threshold_max, validation_min, validation_max, min_threshold_files, max_threshold_files
           
            # ✅ Validate against weighted average range (for NON-ZERO test amounts)
            is_valid = validation_min <= amount <= validation_max
            
            range_msg = f"Range: ₹{validation_min:.2f} - ₹{validation_max:.2f}"
            zero_info = f" (Zeros in training: {stats['zero_percentage']:.1f}%)" if stats['zero_allowed'] else ""
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
        print("COMPREHENSIVE TRAINING SUMMARY (ZEROS EXCLUDED FROM ALL CALCULATIONS)")
        print("="*70)
       
        total_particulars = 0
        total_records = 0
        total_zero_records = 0
       
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            range_particulars = len(range_data['amount_ranges'])
            range_records = sum(stats['total_count'] for stats in range_data['amount_ranges'].values())
            range_zeros = sum(stats['zero_count'] for stats in range_data['amount_ranges'].values())
           
            if range_particulars > 0:
                print(f"\nSq Feet Range: {sqfeet_range}")
                print(f"  Unique particulars: {range_particulars}")
                print(f"  Total records: {range_records}")
                print(f"  Zero amount records: {range_zeros} ({(range_zeros/range_records)*100:.1f}%)")
               
                zero_allowed_count = sum(1 for stats in range_data['amount_ranges'].values() if stats['zero_allowed'])
                print(f"  Particulars allowing zeros: {zero_allowed_count}/{range_particulars}")
               
                # Average threshold range (excluding zeros)
                threshold_ranges = [stats['validation_range_upper'] - stats['validation_range_lower']
                                   for stats in range_data['amount_ranges'].values()]
                
                # Average weighted range (excluding zeros)
                weighted_ranges = [stats['weighted_avg_max'] - stats['weighted_avg_min']
                                   for stats in range_data['amount_ranges'].values() if 'weighted_avg_max' in stats]
               
                if threshold_ranges:
                    print(f"  Average threshold range size (NON-ZERO only): ₹{np.mean(threshold_ranges):.2f}")
                
                if weighted_ranges:
                    print(f"  Average weighted range size (NON-ZERO only): ₹{np.mean(weighted_ranges):.2f}")
               
                # Show top 3 particulars with their zero statistics
                sorted_particulars = sorted(range_data['amount_ranges'].items(),
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
        print(f"Total zero amount records: {total_zero_records} ({(total_zero_records/total_records)*100:.1f}%)")
        print("\n✅ Model is ready - ZEROS EXCLUDED from threshold & weighted avg calculations!")
        print("✅ Zero amounts in test files will be validated based on training data presence.")
        print("="*70)

def main():
    print("CWI Validator - HDFC Amount Training Phase (ZEROS EXCLUDED FROM ALL CALCULATIONS)")
    print("="*80)
   
    validator = CWIValidator()
   
    training_folder = r"C:\Users\Tanush.Bidkar\Downloads\CWI_Validator_1001_1500\training_files"
   
    training_data = validator.load_training_data(training_folder)
   
    if not training_data:
        print("No training data loaded. Please check your file path and data format.")
        print("\nMake sure your Excel filenames contain sq feet ranges like:")
        print("  - 'data_500-1000.xlsx'")
        print("  - 'cwi_1001-1500.xlsx'")
        print("  - 'project_3001-3500.xlsx'")
        return
   
    validator.calculate_amount_ranges(training_data)
    validator.train_text_similarity()
   
    output_folder = r"C:\Users\Tanush.Bidkar\Downloads\CWI_Validator_1001_1500"
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_path = os.path.join(output_folder, "trained_model_sqfeet_zero_aware_logic_amount_HDFC.pkl")
    validator.save_model(model_path)
   
    validator.display_training_summary()
   
    # Generate enhanced detailed report
    report_path = os.path.join(output_folder, "training_report_sqfeet_weighted_avg_amount_ZERO_EXCLUDED_HDFC.txt")
    
    print(f"\nGenerating detailed report at: {report_path}...")
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CWI Validator Enhanced Training Report (ZEROS EXCLUDED FROM ALL CALCULATIONS - AMOUNT)\n")
            f.write("="*80 + "\n\n")
           
            for sqfeet_range, range_data in validator.sqfeet_ranges.items():
                if len(range_data['amount_ranges']) > 0:
                    f.write(f"Sq Feet Range: {sqfeet_range}\n")
                    f.write("-" * 40 + "\n")
                   
                    for particular, stats in range_data['amount_ranges'].items():
                        f.write(f"Particular: {particular}\n")
                        f.write(f"  Total Count: {stats['total_count']}\n")
                        f.write(f"  Zero Count: {stats['zero_count']} ({stats['zero_percentage']:.1f}%)\n")
                        f.write(f"  Non-Zero Count: {stats['non_zero_count']}\n")
                        f.write(f"  Zero Allowed: {stats['zero_allowed']}\n")
                        f.write(f"  Overall Mean (NON-ZERO only): ₹{stats['overall_mean']:.2f}\n")
                        f.write(f"  Overall Std (NON-ZERO only): ₹{stats['overall_std']:.2f}\n")
                        f.write(f"  Overall Min (NON-ZERO only): ₹{stats['overall_min']:.2f}\n")
                        f.write(f"  Overall Max (NON-ZERO only): ₹{stats['overall_max']:.2f}\n")
                        
                        if 'validation_range_lower' in stats:
                            f.write(f"  Threshold Range (NON-ZERO only): ₹{stats['validation_range_lower']:.2f} - ₹{stats['validation_range_upper']:.2f}\n")
                        
                        if 'weighted_avg_min' in stats:
                            f.write(f"  Weighted Avg Range (NON-ZERO only): ₹{stats['weighted_avg_min']:.2f} - ₹{stats['weighted_avg_max']:.2f}\n")
                        
                        # Write files information for Min Amount (NON-ZERO)
                        if stats.get('min_threshold_files'):
                            f.write(f"  Files with Min Amount (NON-ZERO):\n")
                            for mf in stats['min_threshold_files'][:5]:  # Show first 5
                                f.write(f"    - {mf['file']}: {mf['original_particular']} (₹{mf['amount']:.2f})\n")
                            if len(stats['min_threshold_files']) > 5:
                                f.write(f"    ... and {len(stats['min_threshold_files']) - 5} more\n")
                        
                        # Write files information for Max Amount (NON-ZERO)
                        if stats.get('max_threshold_files'):
                            f.write(f"  Files with Max Amount (NON-ZERO):\n")
                            for mf in stats['max_threshold_files'][:5]:  # Show first 5
                                f.write(f"    - {mf['file']}: {mf['original_particular']} (₹{mf['amount']:.2f})\n")
                            if len(stats['max_threshold_files']) > 5:
                                f.write(f"    ... and {len(stats['max_threshold_files']) - 5} more\n")
                        
                        f.write("\n")
                   
                    f.write("\n")
       
        print(f"✓ Enhanced training report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error saving report: {e}")

if __name__ == "__main__":
    main()