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
        # Store data segregated by sq feet ranges
        self.sqfeet_ranges = {
            '500-1000': {'quantity_ranges': {}, 'particulars_data': defaultdict(list)},
            '1001-1500': {'quantity_ranges': {}, 'particulars_data': defaultdict(list)},
            '1501-2000': {'quantity_ranges': {}, 'particulars_data': defaultdict(list)},
            '2001-2500': {'quantity_ranges': {}, 'particulars_data': defaultdict(list)},
            '2501-3000': {'quantity_ranges': {}, 'particulars_data': defaultdict(list)},
            '3001-3500': {'quantity_ranges': {}, 'particulars_data': defaultdict(list)},
            '3501-4000': {'quantity_ranges': {}, 'particulars_data': defaultdict(list)},
            '4001-4500': {'quantity_ranges': {}, 'particulars_data': defaultdict(list)},
            '4501-5000': {'quantity_ranges': {}, 'particulars_data': defaultdict(list)},
            '5000+': {'quantity_ranges': {}, 'particulars_data': defaultdict(list)}
        }
       
        # Text similarity models for each range
        self.vectorizers = {}
        self.particular_vectors = {}
        self.particular_names = {}
       
        self.training_stats = {}
   
    def extract_sqfeet_from_filename(self, filename):
        """Extract square feet range from filename"""
        filename_lower = filename.lower()
       
        # Look for patterns like "500-1000", "1001-1500", etc.
        patterns = [
            r'500\s*-?\s*1000',
            r'1001\s*-?\s*1500',
            r'1501\s*-?\s*2000',
            r'2001\s*-?\s*2500',
            r'2501\s*-?\s*3000',
            r'3001\s*-?\s*3500',
            r'3501\s*-?\s*4000',
            r'4001\s*-?\s*4500',
            r'4501\s*-?\s*5000'
        ]
       
        range_mapping = {
            0: '500-1000',
            1: '1001-1500',
            2: '1501-2000',
            3: '2001-2500',
            4: '2501-3000',
            5: '3001-3500',
            6: '3501-4000',
            7: '4001-4500',
            8: '4501-5000'
        }
       
        for i, pattern in enumerate(patterns):
            if re.search(pattern, filename_lower):
                return range_mapping[i]
       
        # Check for 5000+ pattern
        if re.search(r'5000\+|above\s*5000|more\s*than\s*5000', filename_lower):
            return '5000+'
       
        # Try to extract any number and determine range
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
   
    def extract_quantity(self, qty_value):
        """Extract numeric quantity from various formats"""
        if pd.isna(qty_value):
            return 0
       
        qty_str = str(qty_value).strip()
        numbers = re.findall(r'\d+\.?\d*', qty_str)
        if numbers:
            return float(numbers[0])
        return 0

    def normalize_unit(self, unit_value):
        """Normalize unit values to check if it's number-based"""
        if pd.isna(unit_value):
            return None
    
        unit_str = str(unit_value).strip().upper()
    
        # Number-based units
        number_units = ['NOS', 'NO', 'NOS.', 'NO.', 'Each', 'NUMBER', 'NUMBERS', 'QTY', 'QUANTITY']
    
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
                           'fixture', 'board', 'sheet', 'nos', 'number', 'quantity', 'each']

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
                print("Please ensure filename contains sq feet range (e.g., '1001-1500', '500-1000', etc.)")
                continue
           
            print(f"  Detected sq feet range: {sqfeet_range}")
           
            try:
                df = pd.read_excel(file_path, sheet_name='Extracted Data')
               
                # Expected columns (only 3 now)
                expected_cols = ['Sr.No', 'Particulars', 'As Per CWI (Qty)', 'Unit']
                df_cols = df.columns.tolist()
                col_mapping = {}
               
                for expected_col in expected_cols:
                    found = False
                    for df_col in df_cols:
                        if expected_col.lower().replace('.', '').replace(' ', '') in str(df_col).lower().replace('.', '').replace(' ', ''):
                            col_mapping[expected_col] = df_col
                            found = True
                            break
                    if not found:
                        print(f"Warning: Column '{expected_col}' not found in {file_name}")
               
                if len(col_mapping) < 4:
                    print(f"Skipping {file_name} - missing required columns")
                    print(f"Found columns: {list(col_mapping.keys())}")
                    print(f"Available columns in file: {df_cols}")
                    continue
               
                df_renamed = df.rename(columns={v: k for k, v in col_mapping.items()})
               
                file_records = 0
                for idx, row in df_renamed.iterrows():
                    sr_no = row.get('Sr.No', '')
                    particular = row.get('Particulars', '')
                    qty = row.get('As Per CWI (Qty)', 0)
                    unit = row.get('Unit', '')  # ADD THIS LINE
                   
                    if pd.isna(particular) or str(particular).strip() == '':
                        continue
                   
                    cleaned_particular = self.clean_text(particular)
                    extracted_qty = self.extract_quantity(qty)
                   
                    if cleaned_particular:
                        record = {
                            'file': file_name,
                            'sr_no': sr_no,
                            'particular': cleaned_particular,
                            'original_particular': str(particular),
                            'quantity': extracted_qty,
                            'unit': unit,  # ADD THIS LINE
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
   
    def calculate_quantity_ranges(self, training_data):
        """Calculate quantity ranges for each type of particular within each sq feet range"""
        print("\nCalculating quantity ranges by Sq Feet categories (INCLUDING ZEROS)...")
    
        # Group data by sq feet range and particular
        for record in training_data:
            sqfeet_range = record['sqfeet_range']
            particular = record['particular']
            quantity = record['quantity']

            self.sqfeet_ranges[sqfeet_range]['particulars_data'][particular].append(quantity)
    
        # Calculate comprehensive statistics for each range and particular combination
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            if not range_data['particulars_data']:  # Skip empty ranges
                continue

            print(f"\nProcessing range: {sqfeet_range}")
            particulars_in_range = 0

            for particular, quantities in range_data['particulars_data'].items():
                quantities = np.array(quantities)
                non_zero_qty = quantities[quantities > 0]
                zero_qty = quantities[quantities == 0]

                # Overall statistics (including zeros)
                overall_stats = {
                    'total_count': len(quantities),
                    'zero_count': len(zero_qty),
                    'non_zero_count': len(non_zero_qty),
                    'zero_percentage': (len(zero_qty) / len(quantities)) * 100,
                    'zero_allowed': len(zero_qty) > 0,
                    'overall_min': float(np.min(quantities)),
                    'overall_max': float(np.max(quantities)),
                    'overall_mean': float(np.mean(quantities)),
                    'overall_median': float(np.median(quantities)),
                    'overall_std': float(np.std(quantities))
                }

                # Non-zero statistics (for reference)
                if len(non_zero_qty) > 0:
                    non_zero_stats = {
                        'non_zero_min': float(np.min(non_zero_qty)),
                        'non_zero_max': float(np.max(non_zero_qty)),
                        'non_zero_mean': float(np.mean(non_zero_qty)),
                        'non_zero_median': float(np.median(non_zero_qty)),
                        'non_zero_std': float(np.std(non_zero_qty))
                    }
                    overall_stats.update(non_zero_stats)

                # Calculate validation range using OVERALL statistics (including zeros)
                if overall_stats['overall_std'] > 0:
                    # Use overall mean and std for range calculation
                    lower_bound = max(0, overall_stats['overall_mean'] - 2 * overall_stats['overall_std'])
                    upper_bound = overall_stats['overall_mean'] + 2 * overall_stats['overall_std']

                    # If training data contains zeros, ensure lower bound is 0
                    if overall_stats['zero_allowed']:
                        lower_bound = 0

                    # Ensure bounds cover all training data
                    lower_bound = min(lower_bound, overall_stats['overall_min'])
                    upper_bound = max(upper_bound, overall_stats['overall_max'])
                else:
                    # If std is 0, all values are the same
                    if overall_stats['zero_allowed']:
                        lower_bound = 0
                        upper_bound = overall_stats['overall_max'] * 1.2 if overall_stats['overall_max'] > 0 else 0
                    else:
                        lower_bound = overall_stats['overall_min'] * 0.8
                        upper_bound = overall_stats['overall_max'] * 1.2

                # NEW: Get unit for this particular to check if rounding needed
                unit_for_particular = None
                for record in training_data:
                    if record['sqfeet_range'] == sqfeet_range and record['particular'] == particular:
                        unit_for_particular = record['unit']
                        break

                # NEW: Round thresholds if unit is number-based
                if unit_for_particular and self.is_number_based_unit(unit_for_particular):
                    lower_bound = round(lower_bound)
                    upper_bound = round(upper_bound)

                # NEW: Calculate weighted averages (different from thresholds)
                weighted_avg_min = overall_stats['overall_mean'] - overall_stats['overall_std']
                weighted_avg_max = overall_stats['overall_mean'] + overall_stats['overall_std']

                # Ensure weighted averages are within threshold bounds
                weighted_avg_min = max(lower_bound, weighted_avg_min)
                weighted_avg_max = min(upper_bound, weighted_avg_max)

                # Round if number-based unit
                if unit_for_particular and self.is_number_based_unit(unit_for_particular):
                    weighted_avg_min = round(weighted_avg_min)
                    weighted_avg_max = round(weighted_avg_max)
                # NEW: Get file information for min and max thresholds
                min_threshold_files = []
                max_threshold_files = []

                for record in training_data:
                    if record['sqfeet_range'] == sqfeet_range and record['particular'] == particular:
                        qty = record['quantity']
                        # Track files that have the minimum quantity
                        if qty == overall_stats['overall_min']:
                            min_threshold_files.append({
                                'file': record['file'],
                                'original_particular': record['original_particular'],
                                'quantity': qty
                            })
                        # Track files that have the maximum quantity
                        if qty == overall_stats['overall_max']:
                            max_threshold_files.append({
                                'file': record['file'],
                                'original_particular': record['original_particular'],
                                'quantity': qty
                            })
                overall_stats.update({
                    'validation_range_lower': lower_bound,
                    'validation_range_upper': upper_bound,
                    'weighted_avg_min': weighted_avg_min,
                    'weighted_avg_max': weighted_avg_max,
                    'unit': unit_for_particular,
                    'min_threshold_files': min_threshold_files,  # NEW
                    'max_threshold_files': max_threshold_files # NEW    
                })

                range_data['quantity_ranges'][particular] = overall_stats
                particulars_in_range += 1

            print(f"  ✓ Calculated ranges for {particulars_in_range} unique particulars")
    
        # Display summary
        print(f"\nSummary by Sq Feet Range:")
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            particular_count = len(range_data['quantity_ranges'])
            if particular_count > 0:
                print(f"  {sqfeet_range}: {particular_count} unique particulars")
           
    def train_text_similarity(self):
        """Train text similarity model for each sq feet range"""
        print("\nTraining text similarity models for each Sq Feet range...")
       
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            particular_names = list(range_data['quantity_ranges'].keys())
           
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
   
    def validate_quantity(self, particular, quantity, sqfeet_range):
        """Validate quantity for a particular within specific sq feet range"""
        if sqfeet_range not in self.sqfeet_ranges:
            return False, f"Invalid sq feet range: {sqfeet_range}"
       
        range_data = self.sqfeet_ranges[sqfeet_range]
       
        # First try exact match
        if particular in range_data['quantity_ranges']:
            stats = range_data['quantity_ranges'][particular]
           
            # Special handling for zero quantities
            if quantity == 0 and stats['zero_allowed']:
                return True, f"Zero quantity allowed (appears in {stats['zero_count']}/{stats['total_count']} training cases)"
           
            # Validate against overall range
            if 'validation_range_lower' in stats and 'validation_range_upper' in stats:
                is_valid = stats['validation_range_lower'] <= quantity <= stats['validation_range_upper']
                range_msg = f"Range: {stats['validation_range_lower']:.2f} - {stats['validation_range_upper']:.2f}"
                zero_info = f" (Zeros: {stats['zero_percentage']:.1f}%)" if stats['zero_allowed'] else ""
                return is_valid, f"Exact match - {range_msg}{zero_info}"
       
        # Try similarity matching
        similar_particular, similarity = self.find_similar_particular(particular, sqfeet_range)
        if similar_particular:
            stats = range_data['quantity_ranges'][similar_particular]
           
            # Special handling for zero quantities
            if quantity == 0 and stats['zero_allowed']:
                return True, f"Zero quantity allowed via similar match ({similarity:.2f}) - appears in {stats['zero_count']}/{stats['total_count']} training cases"
           
            # Validate against overall range
            if 'validation_range_lower' in stats and 'validation_range_upper' in stats:
                is_valid = stats['validation_range_lower'] <= quantity <= stats['validation_range_upper']
                range_msg = f"Range: {stats['validation_range_lower']:.2f} - {stats['validation_range_upper']:.2f}"
                zero_info = f" (Zeros: {stats['zero_percentage']:.1f}%)" if stats['zero_allowed'] else ""
                return is_valid, f"Similar match ({similarity:.2f}) - {range_msg}{zero_info}"
       
        return False, "No matching particular found"
   
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
        print("COMPREHENSIVE TRAINING SUMMARY (INCLUDING ZERO ANALYSIS)")
        print("="*70)
       
        total_particulars = 0
        total_records = 0
        total_zero_records = 0
       
        for sqfeet_range, range_data in self.sqfeet_ranges.items():
            range_particulars = len(range_data['quantity_ranges'])
            range_records = sum(stats['total_count'] for stats in range_data['quantity_ranges'].values())
            range_zeros = sum(stats['zero_count'] for stats in range_data['quantity_ranges'].values())
           
            if range_particulars > 0:
                print(f"\nSq Feet Range: {sqfeet_range}")
                print(f"  Unique particulars: {range_particulars}")
                print(f"  Total records: {range_records}")
                print(f"  Zero quantity records: {range_zeros} ({(range_zeros/range_records)*100:.1f}%)")
               
                zero_allowed_count = sum(1 for stats in range_data['quantity_ranges'].values() if stats['zero_allowed'])
                print(f"  Particulars allowing zeros: {zero_allowed_count}/{range_particulars}")
               
                # Average validation range
                ranges = [stats['validation_range_upper'] - stats['validation_range_lower']
                          for stats in range_data['quantity_ranges'].values() if 'validation_range_upper' in stats]
               
                if ranges:
                    print(f"  Average validation range size: {np.mean(ranges):.2f}")
               
                # Show top 3 particulars with their zero statistics
                sorted_particulars = sorted(range_data['quantity_ranges'].items(),
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
        print(f"Total zero quantity records: {total_zero_records} ({(total_zero_records/total_records)*100:.1f}%)")
        print("\nModel is ready for validation with ZERO-AWARE logic!")
        print("="*70)

def main():
    print("CWI Validator - Enhanced Training Phase (Zero-Aware)")
    print("="*60)
   
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
   
    validator.calculate_quantity_ranges(training_data)
    validator.train_text_similarity()
   
    output_folder = r"C:\Users\Tanush.Bidkar\Downloads\CWI_Validator_1001_1500"
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_path = os.path.join(output_folder, "trained_model_sqfeet_zero_aware.pkl")
    validator.save_model(model_path)
   
    validator.display_training_summary()
   
    # Generate enhanced detailed report
    report_path = os.path.join(output_folder, "training_report_sqfeet_unit_zero_aware_HDFC.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CWI Validator Enhanced Training Report (Unit-Aware, Zero-Aware Statistics)\n")
        f.write("="*80 + "\n\n")
        
        for sqfeet_range, range_data in validator.sqfeet_ranges.items():
            if len(range_data['quantity_ranges']) > 0:
                f.write(f"Sq Feet Range: {sqfeet_range}\n")
                f.write("-" * 40 + "\n")
                
                for particular, stats in range_data['quantity_ranges'].items():
                    f.write(f"Particular: {particular}\n")
                    f.write(f"  Unit: {stats.get('unit', 'N/A')}\n")
                    f.write(f"  Total Count: {stats['total_count']}\n")
                    f.write(f"  Zero Count: {stats['zero_count']} ({stats['zero_percentage']:.1f}%)\n")
                    f.write(f"  Non-Zero Count: {stats['non_zero_count']}\n")
                    f.write(f"  Zero Allowed: {stats['zero_allowed']}\n")
                    f.write(f"  Overall Mean: {stats['overall_mean']:.2f}\n")
                    f.write(f"  Overall Std: {stats['overall_std']:.2f}\n")
                    if 'non_zero_mean' in stats:
                        f.write(f"  Non-Zero Mean: {stats['non_zero_mean']:.2f}\n")
                        f.write(f"  Non-Zero Std: {stats['non_zero_std']:.2f}\n")
                    if 'validation_range_lower' in stats:
                        f.write(f"  Validation Range: {stats['validation_range_lower']:.2f} - {stats['validation_range_upper']:.2f}\n")
                    f.write("\n")
                
                f.write("\n")
    
    print(f"✓ Enhanced training report saved to: {report_path}")

if __name__ == "__main__":
    main()