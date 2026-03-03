import os
import win32com.client
import time

# --- CONFIGURATION ---
root_directory = r'C:\Users\Tanush.Bidkar\Downloads\OneDrive_2026-01-30\Reliance Threshold Work\West\Yousta\10371sqft_Nallasopara'

# --- VBA MACRO (WITH ERROR PROTECTION) ---
vba_macro_code = """
SSub ExtractYoustaData_Strict()
    Dim ws As Worksheet, destWs As Worksheet
    Dim searchRange As Range, cwiCell As Range, cbtCell As Range, headerCell As Range
    Dim lastRow As Long, destRow As Long, r As Long
    Dim sourceSheet As Worksheet
    Dim cwiFound As Boolean, cbtFound As Boolean
    Dim startRow As Long
    Dim servCol As Long, descCol As Long, rateCol As Long, uomCol As Long
    Dim qtyCol As Long, amtCol As Long
    Dim cellTxt As String
    Dim rawVal As Variant
    Dim rOffset As Integer, cOffset As Integer
    
    Application.ScreenUpdating = False
    Application.DisplayAlerts = False
    
    ' 1. Delete old "Extracted Data" sheet if it exists
    On Error Resume Next
    ThisWorkbook.Sheets("Extracted Data").Delete
    On Error GoTo 0
    
    ' 2. Create new Destination Sheet
    Set destWs = ThisWorkbook.Sheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
    destWs.Name = "Extracted Data"
    
    ' 3. Set Headers
    destWs.Cells(1, 1).Value = "Sr. No"
    destWs.Cells(1, 2).Value = "Services Codes"
    destWs.Cells(1, 3).Value = "Particulars"
    destWs.Cells(1, 4).Value = "RC Rates"
    destWs.Cells(1, 5).Value = "Units"
    destWs.Cells(1, 6).Value = "As per CWI (Qty)"
    destWs.Cells(1, 7).Value = "As per CWI (Amount)"
    
    destRow = 2
    cwiFound = False
    cbtFound = False
    
    ' 4. Find the Source Sheet (Look for CWI first, then CBT)
    For Each ws In ThisWorkbook.Worksheets
        If ws.Name <> "Extracted Data" Then
            ' Search Area: First 40 rows, columns A to BZ
            Set searchRange = ws.Range("A1:BZ40")
            
            ' Priority 1: CWI
            Set cwiCell = searchRange.Find("CWI", LookIn:=xlValues, LookAt:=xlPart, MatchCase:=False)
            If Not cwiCell Is Nothing Then
                Set sourceSheet = ws
                Set headerCell = cwiCell
                cwiFound = True
                Exit For
            End If
        End If
    Next ws
    
    ' Priority 2: CBT (Fallback)
    If Not cwiFound Then
        For Each ws In ThisWorkbook.Worksheets
            If ws.Name <> "Extracted Data" Then
                Set searchRange = ws.Range("A1:BZ40")
                Set cbtCell = searchRange.Find("CBT", LookIn:=xlValues, LookAt:=xlPart, MatchCase:=False)
                If Not cbtCell Is Nothing Then
                    Set sourceSheet = ws
                    Set headerCell = cbtCell
                    cbtFound = True
                    Exit For
                End If
            End If
        Next ws
    End If
    
    ' 5. Extract Data
    If cwiFound Or cbtFound Then
        With sourceSheet
            Dim headRow As Long
            headRow = headerCell.Row
            
            ' A. Find "Description" or "Particulars" to define the main table structure
            '    We search in the same row as CWI/CBT or up to 5 rows down
            Dim rngDesc As Range
            Set rngDesc = .Range("A" & headRow & ":Z" & headRow + 10).Find("Description", LookIn:=xlValues, LookAt:=xlPart, MatchCase:=False)
            If rngDesc Is Nothing Then Set rngDesc = .Range("A" & headRow & ":Z" & headRow + 10).Find("Particulars", LookIn:=xlValues, LookAt:=xlPart, MatchCase:=False)
            
            If Not rngDesc Is Nothing Then
                descCol = rngDesc.Column
                startRow = rngDesc.Row + 2 ' Data usually starts 2 rows after the header
                
                ' Assume Service Code is to the left of Description
                servCol = descCol - 1
                If servCol < 1 Then servCol = 1
                
                ' Find UOM and Rate (Search in the description header row)
                On Error Resume Next
                uomCol = .Rows(rngDesc.Row).Find("UOM", LookIn:=xlValues, LookAt:=xlPart, MatchCase:=False).Column
                rateCol = .Rows(rngDesc.Row).Find("Rate", LookIn:=xlValues, LookAt:=xlPart, MatchCase:=False).Column
                On Error GoTo 0
                
                ' --- STRICT QTY/AMOUNT FINDER ---
                ' We search relative to the found CWI/CBT header ONLY.
                ' We look 0 to 4 columns to the right of the header, and 0 to 5 rows down.
                ' This prevents grabbing "Qty" from other sections (like 'Approved by NHQ').
                
                qtyCol = 0
                amtCol = 0
                
                For rOffset = 0 To 5
                    ' Strict limit: Only look 4 columns wide starting from the CWI/CBT Header
                    For cOffset = 0 To 4
                        On Error Resume Next
                        rawVal = .Cells(headRow + rOffset, headerCell.Column + cOffset).Value
                        On Error GoTo 0
                        
                        If Not IsError(rawVal) Then
                            cellTxt = LCase(Trim(CStr(rawVal)))
                            
                            ' Qty Logic: Look for "qty" or "quantity"
                            If qtyCol = 0 Then
                                If InStr(1, cellTxt, "qty") > 0 Or InStr(1, cellTxt, "quantity") > 0 Then
                                    qtyCol = headerCell.Column + cOffset
                                End If
                            End If
                            
                            ' Amount Logic: Look for "amount" or "amt"
                            If amtCol = 0 Then
                                If InStr(1, cellTxt, "amount") > 0 Or InStr(1, cellTxt, "amt") > 0 Then
                                    amtCol = headerCell.Column + cOffset
                                End If
                            End If
                        End If
                    Next cOffset
                Next rOffset
                
                ' B. Loop through rows and copy data
                lastRow = .Cells(.Rows.Count, descCol).End(xlUp).Row
                
                For r = startRow To lastRow
                    Dim pVal As String
                    On Error Resume Next
                    pVal = Trim(CStr(.Cells(r, descCol).Value))
                    On Error GoTo 0
                    
                    ' Conditions to copy:
                    ' 1. Particulars is not empty
                    ' 2. Not a "Total" row
                    ' 3. Not a "Rate/Sq.ft" row
                    If pVal <> "" And InStr(1, pVal, "Total", vbTextCompare) = 0 And InStr(1, pVal, "Sub Total", vbTextCompare) = 0 And Left(pVal, 4) <> "Rate" Then
                        
                        destWs.Cells(destRow, 1).Value = destRow - 1 ' Sr No
                        
                        If servCol > 0 Then destWs.Cells(destRow, 2).Value = .Cells(r, servCol).Value
                        destWs.Cells(destRow, 3).Value = .Cells(r, descCol).Value
                        If rateCol > 0 Then destWs.Cells(destRow, 4).Value = .Cells(r, rateCol).Value
                        If uomCol > 0 Then destWs.Cells(destRow, 5).Value = .Cells(r, uomCol).Value
                        
                        ' Copy Qty and Amount
                        If qtyCol > 0 Then destWs.Cells(destRow, 6).Value = .Cells(r, qtyCol).Value
                        If amtCol > 0 Then destWs.Cells(destRow, 7).Value = .Cells(r, amtCol).Value
                        
                        destRow = destRow + 1
                    End If
                Next r
            Else
                MsgBox "Could not find 'Description' or 'Particulars' column.", vbExclamation
            End If
        End With
    Else
        MsgBox "Neither 'CWI' nor 'CBT' headers were found in this workbook.", vbExclamation
    End If
    
    destWs.Columns.AutoFit
    Application.ScreenUpdating = True
    Application.DisplayAlerts = True
    
    MsgBox "Extraction Complete!", vbInformation
End Sub
"""

def cleanup_excel():
    try:
        os.system("taskkill /f /im excel.exe >nul 2>&1")
        time.sleep(1)
    except:
        pass

def process_excel_files():
    print("Initializing Excel...")
    cleanup_excel() 
    
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False 
        excel.AutomationSecurity = 1 
    except Exception as e:
        print(f"❌ Critical Error starting Excel: {e}")
        return

    print(f"Scanning: {root_directory}")
    
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith(('.xlsx', '.xlsm', '.xls')) and not file.startswith("~$"):
                
                file_path = os.path.abspath(os.path.join(root, file))
                print(f"Processing: {file}...")
                
                wb = None
                try:
                    # Open File
                    try:
                        wb = excel.Workbooks.Open(file_path, CorruptLoad=1)
                    except:
                        print("   ⚠️ Normal open failed. Attempting repair mode...")
                        wb = excel.Workbooks.Open(file_path, CorruptLoad=2)

                    # 1. CLEANUP
                    try:
                        old_comp = wb.VBProject.VBComponents("ExtractionMod")
                        wb.VBProject.VBComponents.Remove(old_comp)
                    except:
                        pass 
                    
                    # 2. INJECT MACRO
                    vb_comp = wb.VBProject.VBComponents.Add(1)
                    vb_comp.Name = "ExtractionMod"
                    vb_comp.CodeModule.AddFromString(vba_macro_code)
                    
                    # 3. RUN MACRO
                    excel.Run(f"'{wb.Name}'!ExtractionMod.ExtractDataWithPriority")
                    
                    # 4. REMOVE MACRO
                    try:
                        added_comp = wb.VBProject.VBComponents("ExtractionMod")
                        wb.VBProject.VBComponents.Remove(added_comp)
                    except:
                        pass

                    # 5. CONVERSION LOGIC
                    if file.lower().endswith('.xlsm'):
                        new_path = file_path.replace('.xlsm', '.xlsx')
                        wb.SaveAs(new_path, FileFormat=51, ConflictResolution=2)
                        print(f"   ✅ Converted .xlsm -> .xlsx")
                        wb.Close()
                        
                    elif file.lower().endswith('.xlsx'):
                        wb.Save()
                        print(f"   ✅ Updated .xlsx")
                        wb.Close()

                except Exception as e:
                    print(f"   ❌ FAILED: {e}")
                    if wb: 
                        try: wb.Close(SaveChanges=False) 
                        except: pass

    excel.Quit()
    cleanup_excel()
    print("\n✅ All files processed.")

if __name__ == "__main__":
    process_excel_files()