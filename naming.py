import os
import pandas as pd

# Path from your latest screenshot
root_directory = r'C:\Users\Tanush.Bidkar\Downloads\OneDrive_2026-01-30\Reliance Threshold Work'
output_file = "Store_Summary_Report_Final.xlsx"

def process_and_report(root_path):
    if not os.path.exists(root_path):
        print(f"Error: Path '{root_path}' not found.")
        return

    report_data = []
    
    # 1. Iterate through Zones (East, West, etc.)
    for region in os.listdir(root_path):
        region_path = os.path.join(root_path, region)
        
        if os.path.isdir(region_path):
            print(f"Processing Zone: {region}")
            
            # 2. Iterate through Brands (Yousta, GAP, Smart, etc.)
            for brand in os.listdir(region_path):
                brand_path = os.path.join(region_path, brand)
                
                if os.path.isdir(brand_path):
                    # 3. Iterate through Store folders
                    for store_folder in os.listdir(brand_path):
                        store_path = os.path.join(brand_path, store_folder)
                        
                        if os.path.isdir(store_path):
                            # Extract sqft value from folder name
                            sqft_val = store_folder.split('sqft')[0]
                            prefix = f"{sqft_val}sqft"
                            
                            # Check for any Excel files (now including .xlsm)
                            files_in_folder = os.listdir(store_path)
                            # Added .xlsm to the tuple below
                            excel_files = [f for f in files_in_folder if f.lower().endswith(('.xlsx', '.xls', '.csv', '.xlsm'))]
                            is_file_present = "Yes" if len(excel_files) > 0 else "No"
                            
                            # --- RENAME LOGIC ---
                            for filename in excel_files:
                                # Rename if the file doesn't already start with the sqft prefix
                                if not filename.startswith(prefix):
                                    old_file = os.path.join(store_path, filename)
                                    new_name = f"{prefix}_{filename}"
                                    try:
                                        os.rename(old_file, os.path.join(store_path, new_name))
                                        print(f"Renamed: {filename} -> {new_name}")
                                    except Exception as e:
                                        print(f"Error renaming {filename}: {e}")

                            # --- COLLECT DATA ---
                            report_data.append({
                                "Zone": region,
                                "Store Name": brand,
                                "Sheet Name": store_folder,
                                "Sq.feet": sqft_val,
                                "File Present": is_file_present
                            })

    # CREATE SUMMARY EXCEL
    if report_data:
        df = pd.DataFrame(report_data)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for zone in sorted(df['Zone'].unique()):
                zone_df = df[df['Zone'] == zone].copy()
                
                # Insert Sr. No at the start
                zone_df.insert(0, 'Sr. No', range(1, len(zone_df) + 1))
                
                # Order: Sr. No, Store Name, Sheet Name, Sq.feet, File Present
                zone_df.drop(columns=['Zone']).to_excel(writer, sheet_name=zone, index=False)
        
        print(f"\n✅ Success! Report and renaming complete: {os.path.abspath(output_file)}")
    else:
        print("❌ No data found to process.")

if __name__ == "__main__":
    process_and_report(root_directory)