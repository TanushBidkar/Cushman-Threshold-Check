import pandas as pd

# 1. Define the data exactly as you provided
data = {
    "Name": [
        "Aryan Merchant",
        "Abhay Patwa",
        "Ashish Devlekar"
    ],
    "Email": [
        "Aryan.Merchant@cushwake.com",
        "Abhay.Patwa@cushwake.com",
        "Ashish.Devlekar@cushwake.com"
    ]
}

# 2. Create the list (DataFrame)
df = pd.DataFrame(data)

# 3. Save it as an Excel file
file_name = "contacts.xlsx"
df.to_excel(file_name, index=False)

print(f"Success! '{file_name}' has been created with {len(df)} contacts.")
print("You can now run the main sending script.")