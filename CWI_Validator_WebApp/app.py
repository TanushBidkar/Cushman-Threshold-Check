import os
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for, jsonify
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

# --- Matplotlib Setup (for backend graph generation) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- CWIValidatorTester Class (No changes here) ---
class CWIValidatorTester:
    def __init__(self):
        self.sqfeet_ranges = {}
        self.vectorizers = {}
        self.particular_vectors = {}
        self.particular_names = {}
        self.training_stats = {}

    def extract_sqfeet_from_filename(self, filename):
        filename_lower = filename.lower()
        patterns = [
            r'500\s*-?\s*1000', r'1001\s*-?\s*1500', r'1501\s*-?\s*2000',
            r'2001\s*-?\s*2500', r'2501\s*-?\s*3000', r'3001\s*-?\s*3500',
            r'4001\s*-?\s*4500', r'4501\s*-?\s*5000'
        ]
        range_mapping = {
            0: '500-1000', 1: '1001-1500', 2: '1501-2000', 3: '2001-2500',
            4: '2501-3000', 5: '3001-3500', 6: '3501-4000', 7: '4001-4500',
            8: '4501-5000'
        }
        for i, pattern in enumerate(patterns):
            if re.search(pattern, filename_lower):
                return range_mapping[i]
        if re.search(r'5000\+|above\s*5000|more\s*than\s*5000', filename_lower):
            return '5000+'
        numbers = re.findall(r'\d+', filename_lower)
        if numbers:
            for num_str in numbers:
                try:
                    num = int(num_str)
                    if 500 <= num <= 1000: return '500-1000'
                    elif 1001 <= num <= 1500: return '1001-1500'
                    elif 1501 <= num <= 2000: return '1501-2000'
                    elif 2001 <= num <= 2500: return '2001-2500'
                    elif 2501 <= num <= 3000: return '2501-3000'
                    elif 3001 <= num <= 3500: return '3001-3500'
                    elif 3501 <= num <= 4000: return '3501-4000'
                    elif 4001 <= num <= 4500: return '4001-4500'
                    elif 4501 <= num <= 5000: return '4501-5000'
                    elif num > 5000: return '5000+'
                except ValueError:
                    continue
        return None

    def clean_text(self, text):
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def extract_quantity(self, qty_value):
        if pd.isna(qty_value): return 0
        qty_str = str(qty_value).strip()
        numbers = re.findall(r'\d+\.?\d*', qty_str)
        if numbers: return float(numbers[0])
        return 0

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.sqfeet_ranges = model_data['sqfeet_ranges']
            self.vectorizers = model_data['vectorizers']
            self.particular_vectors = model_data['particular_vectors']
            self.particular_names = model_data['particular_names']
            self.training_stats = model_data.get('training_stats', {})
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def find_similar_particular(self, query_particular, sqfeet_range, threshold=0.5):
        if sqfeet_range not in self.vectorizers: return None, 0
        vectorizer = self.vectorizers[sqfeet_range]
        particular_vectors = self.particular_vectors[sqfeet_range]
        particular_names = self.particular_names[sqfeet_range]
        query_vector = vectorizer.transform([self.clean_text(query_particular)])
        similarities = cosine_similarity(query_vector, particular_vectors)[0]
        best_match_idx = np.argmax(similarities)
        if similarities[best_match_idx] >= threshold:
            return particular_names[best_match_idx], similarities[best_match_idx]
        return None, similarities[best_match_idx]

    def validate_quantity(self, particular, quantity, sqfeet_range):
        if sqfeet_range not in self.sqfeet_ranges or not self.sqfeet_ranges[sqfeet_range]['quantity_ranges']:
            return True, f"No validation rules for sq feet range: {sqfeet_range}", None, None
        
        range_data = self.sqfeet_ranges[sqfeet_range]
        cleaned_particular = self.clean_text(particular)
        
        if cleaned_particular in range_data['quantity_ranges']:
            stats = range_data['quantity_ranges'][cleaned_particular]
            if quantity == 0 and stats.get('zero_allowed', False):
                return True, "Zero quantity allowed", cleaned_particular, 1.0
            if 'validation_range_lower' in stats:
                is_valid = stats['validation_range_lower'] <= quantity <= stats['validation_range_upper']
                msg = f"Range: {stats['validation_range_lower']:.2f} - {stats['validation_range_upper']:.2f}"
                return is_valid, msg, cleaned_particular, 1.0

        similar_particular, similarity = self.find_similar_particular(particular, sqfeet_range)
        if similar_particular:
            stats = range_data['quantity_ranges'][similar_particular]
            if quantity == 0 and stats.get('zero_allowed', False):
                return True, "Zero quantity allowed (Similar Match)", similar_particular, similarity
            if 'validation_range_lower' in stats:
                is_valid = stats['validation_range_lower'] <= quantity <= stats['validation_range_upper']
                msg = f"Range (from similar match): {stats['validation_range_lower']:.2f} - {stats['validation_range_upper']:.2f}"
                return is_valid, msg, similar_particular, similarity

        return True, "No matching particular found - automatically approved", None, 0

    def validate_test_file(self, test_file_path):
        sqfeet_range = self.extract_sqfeet_from_filename(os.path.basename(test_file_path))
        if not sqfeet_range or sqfeet_range not in self.sqfeet_ranges:
            return None, "Could not determine Sq. Feet range from filename or no rules for this range."

        try:
            df = pd.read_excel(test_file_path, sheet_name='Extracted Data')
            expected_cols = ['Sr.No', 'Particulars', 'As Per CWI (Qty)']
            col_mapping = {}
            for expected_col in expected_cols:
                found = False
                for df_col in df.columns:
                    clean_expected = expected_col.lower().replace('.', '').replace(' ', '')
                    clean_df_col = str(df_col).lower().replace('.', '').replace(' ', '')
                    if clean_expected in clean_df_col:
                        col_mapping[expected_col] = df_col
                        found = True
                        break
                if not found:
                    return None, f"Required column '{expected_col}' could not be found in the uploaded file."

            df_renamed = df.rename(columns={v: k for k, v in col_mapping.items()})
            results = []
            for _, row in df_renamed.iterrows():
                if pd.isna(row.get('Particulars')) or str(row.get('Particulars')).strip() == '': continue
                particular_val = row.get('Particulars')
                qty_val = row.get('As Per CWI (Qty)', 0)
                qty = self.extract_quantity(qty_val)
                is_valid, msg, matched, sim = self.validate_quantity(particular_val, qty, sqfeet_range)
                results.append({
                    'Sr.No': row.get('Sr.No', ''), 'Original_Particular': str(particular_val),
                    'Quantity': qty, 'Sq_Feet_Range': sqfeet_range, 'Is_Valid': is_valid,
                    'Validation_Message': msg, 'Matched_Particular': matched, 'Similarity_Score': sim
                })
            return results, None
        except Exception as e:
            return None, f"An unexpected error occurred while processing the Excel file: {e}"

# --- Flask Application Setup ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'super-secret-key'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_PATH = "trained_model_sqfeet_zero_aware.pkl"
VALIDATOR = CWIValidatorTester()
if os.path.exists(MODEL_PATH):
    VALIDATOR.load_model(MODEL_PATH)
    print("✅ CWI Validator Model loaded successfully.")
else:
    print(f"❌ FATAL ERROR: Model file not found at '{MODEL_PATH}'!")
    VALIDATOR = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_thresholds(message):
    if pd.isna(message): return "NA", "NA"
    range_match = re.search(r'Range: ([\d.]+)\s*-\s*([\d.]+)', str(message))
    if range_match: return float(range_match.group(1)), float(range_match.group(2))
    if "Zero quantity allowed" in str(message): return 0, 0
    return "NA", "NA"

def highlight_invalid(row):
    color = '#FFC7CE'
    style = f'background-color: {color}'
    styles = [''] * len(row)
    if not row['Is_Valid']:
        styles[row.index.get_loc('Quantity')] = style
        styles[row.index.get_loc('Is_Valid')] = style
    return styles

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if not VALIDATOR:
        return render_template('error.html', message="The validation model is not available.")

    if request.method == 'POST':
        if 'file' not in request.files: flash('No file part'); return redirect(request.url)
        file = request.files['file']
        if file.filename == '': flash('No selected file'); return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            results_data, error_msg = VALIDATOR.validate_test_file(filepath)
            if error_msg: flash(error_msg); return redirect(request.url)

            results_df = pd.DataFrame(results_data)
            results_df['Threshold_Min'], results_df['Threshold_Max'] = zip(*results_df['Validation_Message'].map(extract_thresholds))

            # --- SAVE REPORTS ---
            full_report_filename = f"validated_{filename}"
            full_report_filepath = os.path.join(app.config['OUTPUT_FOLDER'], full_report_filename)
            styled_df = results_df.style.apply(highlight_invalid, axis=1)
            styled_df.to_excel(full_report_filepath, engine='xlsxwriter', index=False)
            
            invalid_df = results_df[results_df['Is_Valid'] == False].copy()
            invalid_report_filename = f"invalid_rows_{filename}"
            if not invalid_df.empty:
                invalid_report_filepath = os.path.join(app.config['OUTPUT_FOLDER'], invalid_report_filename)
                invalid_df.to_excel(invalid_report_filepath, index=False)

            # --- PREPARE FOR WEB DISPLAY ---
            display_df = results_df.copy()
            
            def create_analysis_button(row):
                if not row['Is_Valid'] and row['Threshold_Min'] != 'NA':
                    safe_particular = html.escape(row["Original_Particular"])
                    return (f'<button class="btn btn-sm btn-info btn-analysis" '
                            f'data-particular="{safe_particular}" '
                            f'data-quantity="{row["Quantity"]}" '
                            f'data-range="{row["Sq_Feet_Range"]}">'
                            f'View Analysis</button>')
                return ''
            display_df['Analysis'] = display_df.apply(create_analysis_button, axis=1)
            
            display_df['Is_Valid'] = display_df['Is_Valid'].apply(lambda x: '✅ Valid' if x else '❌ Invalid')
            results_html = display_df.to_html(classes='table table-striped table-hover', justify='left', index=False, escape=False)
            
            total = len(results_df)
            valid = results_df['Is_Valid'].sum()
            invalid = total - valid
            accuracy = (valid / total * 100) if total > 0 else 0

            return render_template('results.html', 
                                   table_html=results_html, 
                                   full_report_filename=full_report_filename,
                                   invalid_report_filename=invalid_report_filename,
                                   total=total, valid=valid, invalid=invalid, 
                                   accuracy=f"{accuracy:.2f}%")
        else:
            flash('Invalid file type. Please upload an .xlsx or .xls file.'); return redirect(request.url)

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

        # --- 2. Find statistics for the item from the model ---
        cleaned_particular = VALIDATOR.clean_text(particular)
        matched_particular = None
        if cleaned_particular in VALIDATOR.sqfeet_ranges[sq_range]['quantity_ranges']:
            matched_particular = cleaned_particular
        else:
            sim_p, _ = VALIDATOR.find_similar_particular(particular, sq_range)
            if sim_p: matched_particular = sim_p

        if not matched_particular:
            return jsonify({'error': 'Could not find statistics for this item.'})

        stats = VALIDATOR.sqfeet_ranges[sq_range]['quantity_ranges'][matched_particular]
        mean = stats.get('overall_mean', 0)
        std = stats.get('overall_std', 0)
        lower_bound = stats.get('validation_range_lower', 0)
        upper_bound = stats.get('validation_range_upper', 0)

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
            ax.set_xlabel('Quantity'); ax.set_ylabel('Likelihood')
            explanation = (f"This bell curve shows the expected distribution. The average is <b>{mean:.2f}</b>, and the valid range is from <b>{lower_bound:.2f}</b> to <b>{upper_bound:.2f}</b>. "
                           f"Your value of <b style='color:red;'>{user_value:.2f}</b> is outside this range.")

        elif chart_type == 'bar_plot':
            ax.bar(['Min', 'Max'], [lower_bound, upper_bound], color='green', alpha=0.6, label='Acceptable Range')
            ax.bar(['Your Value'], [user_value], color='red', label='Your Value')
            ax.set_title('Bar Plot Comparison')
            ax.set_ylabel('Quantity')
            explanation = (f"This bar plot compares your value against the minimum and maximum acceptable quantities. "
                           f"Your value is <b style='color:red;'>{user_value:.2f}</b>, while the valid range is <b>{lower_bound:.2f}</b> - <b>{upper_bound:.2f}</b>.")

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
            ax.grid(True, axis='x', linestyle='--', alpha=0.7) # Add grid to X-axis
            ax.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_title('Threshold Line Chart')
            plt.tight_layout()
            explanation = (f"The green line shows the acceptable range from <b>{lower_bound:.2f}</b> to <b>{upper_bound:.2f}</b>. "
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


if __name__ == '__main__':
    app.run(debug=True)