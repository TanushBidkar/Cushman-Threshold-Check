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
        # Updated specific ranges for Reliance
        self.sqfeet_ranges = {
            '1000-3999': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)},
            '4000-4999': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)},
            '5000-5999': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)},
            '6000-6999': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)},
            '7000-7999': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)},
            '8000-8999': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)},
            '9000-10500': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)},
            '10501-12999': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)},
            '13000-14999': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)},
            '15000-19999': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)},
            '20000+': {'quantity_ranges': {}, 'service_code_data': defaultdict(list)}
        }
    
        # Text similarity models for each range
        self.vectorizers = {}
        self.service_code_vectors = {}
        self.service_code_names = {}
       
        self.training_stats = {}
   
    def extract_sqfeet_from_filename(self, filename):
        filename_lower = filename.lower()
        numbers = re.findall(r'\d+', filename_lower)
        if numbers:
            # Take the first large number found in the filename (usually the sqft)
            for num_str in numbers:
                num = int(num_str)
                if num < 1000: continue # Skip small numbers like dates or counts

                if 1000 <= num <= 3999:
                    return '1000-3999'
                elif 4000 <= num <= 4999:
                    return '4000-4999'
                elif 5000 <= num <= 5999:
                    return '5000-5999'
                elif 6000 <= num <= 6999:
                    return '6000-6999'
                elif 7000 <= num <= 7999:
                    return '7000-7999'
                elif 8000 <= num <= 8999:
                    return '8000-8999'
                elif 9000 <= num <= 10500:
                    return '9000-10500'
                elif 10501 <= num <= 12999:
                    return '10501-12999'
                elif 13000 <= num <= 14999:
                    return '13000-14999'
                elif 15000 <= num <= 19999:
                    return '15000-19999'
                elif num >= 20000:
                    return '20000+'
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
        number_units = ['NOS', 'NO', 'NOS.', 'NO.', 'EACH', 'NUMBER', 'NUMBERS', 'QTY', 'QUANTITY']

        for nu in number_units:
            if nu in unit_str:
                return 'NUMBER_BASED'

        return unit_str

    def is_number_based_unit(self, unit_value):
        """Check if unit is number-based"""
        normalized = self.normalize_unit(unit_value)
        return normalized == 'NUMBER_BASED'

    def round_quantity(self, quantity, unit_value):
        """Round quantity appropriately based on unit type"""
        if self.is_number_based_unit(unit_value):
            # Round to nearest integer for number-based units
            return round(quantity)
        else:
            # Round to 2 decimal places for other units
            return round(quantity, 2)
   
    def load_training_data(self, base_folder):
        """Load all Excel files from Yousta folder across all zones (East, West, North, South)"""
        training_data = []
       
        print(f"Loading training data from: {base_folder}")
       
        if not os.path.exists(base_folder):
            print(f"Error: Folder {base_folder} does not exist!")
            return training_data
       
        # Navigate through zones
        zones = ['East', 'West', 'North', 'South']
        
        for zone in zones:
            zone_path = os.path.join(base_folder, zone)
            if not os.path.exists(zone_path):
                print(f"Warning: Zone folder {zone} not found, skipping...")
                continue
            
            # Look for Yousta folder in each zone
            yousta_path = os.path.join(zone_path, 'Yousta')
            if not os.path.exists(yousta_path):
                print(f"Warning: Yousta folder not found in {zone}, skipping...")
                continue
            
            print(f"\nProcessing Zone: {zone}")
            
            # Get all subfolders in Yousta (e.g., 10371sqft_Nallasopara)
            for project_folder in os.listdir(yousta_path):
                project_path = os.path.join(yousta_path, project_folder)
                
                if not os.path.isdir(project_path):
                    continue
                
                # Extract sq feet from folder name
                sqfeet_range = self.extract_sqfeet_from_filename(project_folder)
                if sqfeet_range is None:
                    print(f"  Warning: Could not extract sq feet range from: {project_folder}")
                    continue
                
                print(f"  Project: {project_folder} → Range: {sqfeet_range}")
                
                # Process all Excel files in this project folder
                excel_files = [f for f in os.listdir(project_path) if f.endswith(('.xlsx', '.xls'))]
                
                for file_name in excel_files:
                    file_path = os.path.join(project_path, file_name)
                    print(f"    Processing: {file_name}")
                    
                    try:
                        df = pd.read_excel(file_path, sheet_name='Extracted Data')
                        
                        # Expected columns
                        expected_cols = ['Sr.No', 'Services Codes', 'Particulars', 'Units', 'As Per CWI (Qty)']
                        df_cols = df.columns.tolist()
                        col_mapping = {}
                        
                        for expected_col in expected_cols:
                            found = False
                            for df_col in df_cols:
                                col_compare_expected = expected_col.lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '')
                                col_compare_df = str(df_col).lower().replace('.', '').replace(' ', '').replace('(', '').replace(')', '')
                                
                                if col_compare_expected in col_compare_df or col_compare_df in col_compare_expected:
                                    col_mapping[expected_col] = df_col
                                    found = True
                                    break
                            
                            if not found:
                                print(f"      Warning: Column '{expected_col}' not found in {file_name}")
                        
                        if len(col_mapping) < 4:  # At least Service Code, Units, and Qty needed
                            print(f"      Skipping {file_name} - missing required columns")
                            continue
                        
                        df_renamed = df.rename(columns={v: k for k, v in col_mapping.items()})
                        
                        file_records = 0
                        ignored_zeros = 0
                        
                        for idx, row in df_renamed.iterrows():
                            sr_no = row.get('Sr.No', '')
                            service_code = row.get('Services Codes', '')
                            particular = row.get('Particulars', '')
                            unit = row.get('Units', '')
                            qty = row.get('As Per CWI (Qty)', 0)
                            
                            # Skip if service code is missing
                            if pd.isna(service_code) or str(service_code).strip() == '':
                                continue
                            
                            extracted_qty = self.extract_quantity(qty)
                            
                            # IGNORE ZERO VALUES
                            if extracted_qty == 0:
                                ignored_zeros += 1
                                continue
                            
                            # Round quantity based on unit type
                            extracted_qty = self.round_quantity(extracted_qty, unit)
                            
                            try:
                                # Convert to float first, then int, then string to remove any .0
                                service_code_str = str(int(float(service_code))).strip()
                            except (ValueError, TypeError):
                                # Fallback for non-numeric codes
                                service_code_str = str(service_code).strip()
                            
                            record = {
                                'file': file_name,
                                'zone': zone,
                                'project': project_folder,
                                'sr_no': sr_no,
                                'service_code': service_code_str,
                                'particular': str(particular),
                                'quantity': extracted_qty,
                                'unit': unit,
                                'sqfeet_range': sqfeet_range
                            }
                            training_data.append(record)
                            file_records += 1
                        
                        print(f"      ✓ Loaded {file_records} records (Ignored {ignored_zeros} zero values)")
                        
                    except Exception as e:
                        print(f"      Error processing {file_name}: {str(e)}")
                        continue
       
        print(f"\nTotal training records loaded (excluding zeros): {len(training_data)}")
       
        # Display distribution by sq feet range
        range_counts = {}
        for record in training_data:
            range_key = record['sqfeet_range']
            range_counts[range_key] = range_counts.get(range_key, 0) + 1
       
        print("\nDistribution by Sq Feet Range:")
        for range_key in ['1K-3K', '3K-5K', '5K-8K', '8K-12K', '12K-15K', '15K-20K', '20K-25K', '25K+']:
            count = range_counts.get(range_key, 0)
            if count > 0:
                print(f"  {range_key}: {count} records")
       
        return training_data
   
    def calculate_quantity_ranges(self, training_data):
        """Calculate quantity ranges SEGREGATED by specific square feet ranges"""
        print("\nCalculating segregated quantity ranges by Service Code and Area...")
        
        # 1. Group data by SQFT Range first, then by Service Code
        # Structure: segregated_data[range_name][service_code] = list of records
        segregated_data = defaultdict(lambda: defaultdict(list))
        
        for record in training_data:
            rng = record['sqfeet_range']
            code = record['service_code']
            segregated_data[rng][code].append(record)
    
        print(f"Found data across {len(segregated_data)} square feet ranges.")
    
        # 2. Process each square feet range independently
        for rng_name in self.sqfeet_ranges.keys():
            if rng_name not in segregated_data:
                print(f"  ⚠ No training data found for range: {rng_name}")
                continue
                
            print(f"  Processing range: {rng_name} ({len(segregated_data[rng_name])} service codes)")
            range_specific_stats = {}
            
            for service_code, data_list in segregated_data[rng_name].items():
                quantities = np.array([item['quantity'] for item in data_list])
                unit = data_list[0]['unit']
                
                # Basic Stats for this specific range only
                mean_qty = float(np.mean(quantities))
                std_qty = float(np.std(quantities))
                min_qty = float(np.min(quantities))
                max_qty = float(np.max(quantities))
                
                # Weighted average range (Mean ± 1*Std) clamped to actual min/max in this range
                if std_qty > 0:
                    weighted_avg_min = max(min_qty, mean_qty - std_qty)
                    weighted_avg_max = min(max_qty, mean_qty + std_qty)
                    # Threshold range (Mean ± 2*Std)
                    threshold_min = max(min_qty, mean_qty - 2 * std_qty)
                    threshold_max = min(max_qty, mean_qty + 2 * std_qty)
                else:
                    weighted_avg_min = weighted_avg_max = threshold_min = threshold_max = mean_qty
    
                # Round values based on unit
                if self.is_number_based_unit(unit):
                    weighted_avg_min, weighted_avg_max = round(weighted_avg_min), round(weighted_avg_max)
                    threshold_min, threshold_max = round(threshold_min), round(threshold_max)
                else:
                    weighted_avg_min, weighted_avg_max = round(weighted_avg_min, 2), round(weighted_avg_max, 2)
                    threshold_min, threshold_max = round(threshold_min, 2), round(threshold_max, 2)
    
                # Store stats ONLY for this service code WITHIN this range
                range_specific_stats[service_code] = {
                    'total_count': len(quantities),
                    'mean': mean_qty,
                    'std': std_qty,
                    'min': min_qty,
                    'max': max_qty,
                    'weighted_avg_min': weighted_avg_min,
                    'weighted_avg_max': weighted_avg_max,
                    'threshold_min': threshold_min,
                    'threshold_max': threshold_max,
                    'unit': unit,
                    'particulars': list(set([item['particular'] for item in data_list])),
                    # These file lists will now only contain files belonging to this range
                    'min_files': [item for item in data_list if abs(item['quantity'] - min_qty) < 0.01],
                    'max_files': [item for item in data_list if abs(item['quantity'] - max_qty) < 0.01],
                    'zones': list(set([item['zone'] for item in data_list]))
                }
            
            # Save range-specific data to the main object
            self.sqfeet_ranges[rng_name]['quantity_ranges'] = range_specific_stats
    
        print(f"✓ Calculation complete. Data is now segregated by Area Range.")
        
    def train_text_similarity(self):
        """Train text similarity model for service codes"""
        print("\nTraining text similarity models for Service Codes...")
        
       
        first_range = list(self.sqfeet_ranges.keys())[0]
        service_codes = list(self.sqfeet_ranges[first_range]['quantity_ranges'].keys())
        
        if service_codes:
            # Convert service codes to strings for vectorization
            service_code_strings = [str(sc) for sc in service_codes]
            
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
            service_code_vectors = vectorizer.fit_transform(service_code_strings)
            
            # Store in all ranges
            for sqfeet_range in self.sqfeet_ranges.keys():
                self.vectorizers[sqfeet_range] = vectorizer
                self.service_code_vectors[sqfeet_range] = service_code_vectors
                self.service_code_names[sqfeet_range] = service_codes
            
            print(f"  ✓ Trained with {len(service_codes)} service codes (PAN-INDIA)")
        else:
            print("  ⚠ No service codes found for training")

    def find_similar_service_code(self, query_service_code, sqfeet_range, threshold=0.7):
        """Find most similar service code from training data"""
        if sqfeet_range not in self.vectorizers or sqfeet_range not in self.service_code_vectors:
            return None, 0
        
        vectorizer = self.vectorizers[sqfeet_range]
        service_code_vectors = self.service_code_vectors[sqfeet_range]
        service_code_names = self.service_code_names[sqfeet_range]
        
        query_str = str(query_service_code).strip()
        query_vector = vectorizer.transform([query_str])
        
        similarities = cosine_similarity(query_vector, service_code_vectors)[0]
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity >= threshold:
            return service_code_names[best_match_idx], best_similarity
        
        return None, best_similarity
   
    def validate_quantity(self, service_code, quantity, unit, sqfeet_range):
        """Validate quantity using weighted average ranges"""
        if sqfeet_range not in self.sqfeet_ranges:
            return False, f"Invalid sq feet range: {sqfeet_range}", None, None, None, None, None, None, None
        
        range_data = self.sqfeet_ranges[sqfeet_range]
        
        # Ignore zero quantities during validation
        if quantity == 0:
            return True, "Zero quantity - skipped validation", None, None, None, None, None, None, None
        
        # First try exact match
        try:
            service_code_str = str(int(float(service_code))).strip()
        except (ValueError, TypeError):
            service_code_str = str(service_code).strip()
        
        if service_code_str in range_data['quantity_ranges']:
            stats = range_data['quantity_ranges'][service_code_str]
            
            validation_min = stats['weighted_avg_min']
            validation_max = stats['weighted_avg_max']
            threshold_min = stats['threshold_min']
            threshold_max = stats['threshold_max']
            
            stored_unit = stats['unit']
            min_files = stats.get('min_files', [])
            max_files = stats.get('max_files', [])
            particulars = stats.get('particulars', [])
            
            # Validate quantity
            if self.is_number_based_unit(stored_unit):
                is_valid = validation_min <= round(quantity) <= validation_max
                range_msg = f"Range: {int(validation_min)} - {int(validation_max)}"
            else:
                is_valid = validation_min <= quantity <= validation_max
                range_msg = f"Range: {validation_min:.2f} - {validation_max:.2f}"
            
            return is_valid, f"Exact match - {range_msg}", threshold_min, threshold_max, validation_min, validation_max, min_files, max_files, particulars
        
        # Try similarity matching
        similar_service_code, similarity = self.find_similar_service_code(service_code_str, sqfeet_range)
        
        if similar_service_code:
            stats = range_data['quantity_ranges'][similar_service_code]
            
            validation_min = stats['weighted_avg_min']
            validation_max = stats['weighted_avg_max']
            threshold_min = stats['threshold_min']
            threshold_max = stats['threshold_max']
            
            stored_unit = stats['unit']
            min_files = stats.get('min_files', [])
            max_files = stats.get('max_files', [])
            particulars = stats.get('particulars', [])
            
            # Validate quantity
            if self.is_number_based_unit(stored_unit):
                is_valid = validation_min <= round(quantity) <= validation_max
                range_msg = f"Range: {int(validation_min)} - {int(validation_max)}"
            else:
                is_valid = validation_min <= quantity <= validation_max
                range_msg = f"Range: {validation_min:.2f} - {validation_max:.2f}"
            
            return is_valid, f"Similar match ({similarity:.2f}) - {range_msg}", threshold_min, threshold_max, validation_min, validation_max, min_files, max_files, particulars
        
        return False, "No matching service code found", None, None, None, None, None, None, None
    
    def save_model(self, model_path):
        """Save trained model to file"""
        model_data = {
            'sqfeet_ranges': self.sqfeet_ranges,
            'vectorizers': self.vectorizers,
            'service_code_vectors': self.service_code_vectors,
            'service_code_names': self.service_code_names,
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
        self.service_code_vectors = model_data['service_code_vectors']
        self.service_code_names = model_data['service_code_names']
        self.training_stats = model_data['training_stats']
        
        print(f"✓ Model loaded from: {model_path}")

    def display_training_summary(self):
        """Display comprehensive training summary"""
        print("\n" + "="*80)
        print("PAN-INDIA TRAINING SUMMARY (Service Code Based, Zero Values Excluded)")
        print("="*80)
        
        # Get stats from any range (they're all the same)
        # Get stats from the first range (PAN-INDIA logic makes them identical)
        first_range = list(self.sqfeet_ranges.keys())[0]
        stats_dict = self.sqfeet_ranges[first_range]['quantity_ranges']
        
        print(f"\nTotal Unique Service Codes: {len(stats_dict)}")
        print(f"Training Approach: PAN-INDIA weighted average (all zones combined)")
        print(f"Validation Range: Mean ± 1 Standard Deviation")
        print(f"Zero Values: Excluded from training and validation")
        
        # Show sample service codes
        print(f"\nSample Service Codes (First 10):")
        for i, (service_code, stats) in enumerate(list(stats_dict.items())[:10]):
            print(f"\n{i+1}. Service Code: {service_code}")
            print(f"   Count: {stats['total_count']} | Unit: {stats['unit']}")
            print(f"   Weighted Range: {stats['weighted_avg_min']} - {stats['weighted_avg_max']}")
            print(f"   Zones: {', '.join(stats['zones'])}")
            print(f"   Particulars: {', '.join(stats['particulars'][:2])}...")
        
        print("\n" + "="*80)
        print("Model ready for validation!")
        print("="*80)

def main():
    print("CWI Validator - Service Code Based Training (PAN-INDIA, Zero Values Excluded)")
    print("="*80)
    
    validator = CWIValidator()
    
    # Base folder containing East, West, North, South
    base_folder = r"C:\Users\Tanush.Bidkar\Downloads\OneDrive_2026-01-30\Reliance Threshold Work"
    
    training_data = validator.load_training_data(base_folder)
    
    
    
    if not training_data:
        print("No training data loaded. Please check your folder structure.")
        print("\nExpected structure:")
        print("  Base Folder/")
        print("    East/Yousta/[project folders with sqft]/[excel files]")
        print("    West/Yousta/[project folders with sqft]/[excel files]")
        print("    North/Yousta/[project folders with sqft]/[excel files]")
        print("    South/Yousta/[project folders with sqft]/[excel files]")
        return
    
    validator.calculate_quantity_ranges(training_data)
    validator.train_text_similarity()
    
    output_folder = r"C:\Users\Tanush.Bidkar\Downloads\OneDrive_2026-01-30\Reliance Threshold Work"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_path = os.path.join(output_folder, "trained_model_service_code_pan_india.pkl")
    validator.save_model(model_path)
    
    validator.display_training_summary()
    
    # Generate detailed report
    report_path = os.path.join(output_folder, "training_report_service_code_pan_india.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CWI Validator Training Report - Service Code Based (PAN-INDIA)\n")
        f.write("="*80 + "\n\n")
        f.write("Training Approach: PAN-INDIA weighted average across all zones\n")
        f.write("Zero Values: Excluded from training and validation\n")
        f.write("Decimal Precision: 2 decimal places (0 for number-based units)\n\n")
        
        # Use dynamic lookup for report generation
        first_range = list(validator.sqfeet_ranges.keys())[0]
        stats_dict = validator.sqfeet_ranges[first_range]['quantity_ranges']
        
        for service_code, stats in sorted(stats_dict.items()):
            f.write(f"Service Code: {service_code}\n")
            f.write(f"  Count: {stats['total_count']}\n")
            f.write(f"  Unit: {stats['unit']}\n")
            f.write(f"  Mean: {stats['mean']:.2f}\n")
            f.write(f"  Std Dev: {stats['std']:.2f}\n")
            f.write(f"  Min: {stats['min']:.2f} | Max: {stats['max']:.2f}\n")
            f.write(f"  Weighted Avg Range: {stats['weighted_avg_min']} - {stats['weighted_avg_max']}\n")
            f.write(f"  Threshold Range: {stats['threshold_min']} - {stats['threshold_max']}\n")
            f.write(f"  Zones: {', '.join(stats['zones'])}\n")
            f.write(f"  Sq Feet Ranges: {', '.join(stats['sqfeet_ranges'])}\n")
            f.write(f"  Particulars:\n")
            for particular in stats['particulars'][:5]:
                f.write(f"    - {particular}\n")
            if len(stats['particulars']) > 5:
                f.write(f"    ... and {len(stats['particulars']) - 5} more\n")
            
            if stats['min_files']:
                f.write(f"  Min Quantity Files:\n")
                for mf in stats['min_files'][:3]:
                    f.write(f"    - {mf['zone']}/{mf['project']}/{mf['file']} (Qty: {mf['quantity']})\n")
            
            if stats['max_files']:
                f.write(f"  Max Quantity Files:\n")
                for mf in stats['max_files'][:3]:
                    f.write(f"    - {mf['zone']}/{mf['project']}/{mf['file']} (Qty: {mf['quantity']})\n")
            
            f.write("\n")
    
    print(f"✓ Training report saved to: {report_path}")

if __name__ == "__main__":
    main()