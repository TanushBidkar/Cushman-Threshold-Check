import sys
import os.path as ospath
import time
import re
import os

# Import win32com
try:
    import win32com.client
except ImportError:
    print("ERROR: win32com not installed. Run: pip install pywin32")
    sys.exit(1)

# --- CONFIGURATION ---
# UPDATE THIS PATH to your actual folder
FOLDER_PATH = r"C:\Users\Tanush.Bidkar\Downloads\ICICI AI Files for presentation\Threshold"

# --- OPTIMIZED VBA TEMPLATE (Updated with Rate per Sq.Ft Column + Sub Category Standardization) ---
VBA_MACRO_OPTIMIZED = """
Option Explicit

'#########################################################################################
'# MAIN MACRO: SR.NO + SUB-CATEGORY MAPPING + RATE PER SQFT + SUB CATEGORY STANDARDIZATION
'# UPDATED: Adds "Rate per sq.feet" column in Extracted Data + Standardizes Sub Category names
'#########################################################################################
Sub ExtractCWI_Data(areaInput As Double)
    '--- Settings ---
    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual
    On Error GoTo ErrorHandler

    '--- Variables ---
    Dim targetSheet As Worksheet, ws As Worksheet
    Dim srNoCol As Long, particularsCol As Long, unitCol As Long, subCatCol As Long
    Dim cwiQtyCol As Long, cwiAmtCol As Long, rateCol As Long
    Dim cwiHeaderRow As Long, cwiHeaderCol As Long
    Dim startRow As Long, r As Long, c As Long, i As Long
    Dim finalMsg As String
    
    '--- Variables for Mapping ---
    Dim currentNatureOfWork As String
    Dim sourceSrNo As String
    Dim sourceSubCat As String
    Dim subCatClean As String
    Dim partVal As String, partUpper As String
    
    '--- Collection for BOQ sheets ---
    Dim boqSheets As New Collection
    
    '--- Sheet Detection ---
    ' FIX: Uses ActiveWorkbook to look at the OPENED file
    For Each ws In ActiveWorkbook.Worksheets
        Dim wsNameLower As String
        wsNameLower = LCase(ws.Name)
        
        ' Ignore existing summary/extracted sheets
        If Not (wsNameLower Like "*summary*" Or wsNameLower = "extracted data") Then
            
            ' UPDATED SHEET PATTERNS:
            ' *boq* covers: BOQ, BOQ1, BOQ2, Consolidated BOQ, Post Audit BOQ
            ' *bill* covers: Bill, Consolidated Bill
            If wsNameLower Like "*boq*" Or _
               wsNameLower Like "*bill*" Or _
               wsNameLower Like "*measurement*" Or _
               wsNameLower Like "*cwi working*" Or _
               wsNameLower Like "*abstract*" Or _
               wsNameLower Like "*estimate*" Then
               
               boqSheets.Add ws
            End If
        End If
    Next ws

    If boqSheets.Count = 0 Then
        ' If no recognized sheet found, default to the first sheet
        boqSheets.Add ActiveWorkbook.Worksheets(1)
    End If

    ' Step 1: Create/Reset "Extracted Data" Sheet
    On Error Resume Next
    Application.DisplayAlerts = False
    ActiveWorkbook.Sheets("Extracted Data").Delete
    ActiveWorkbook.Sheets("Summary Sheet").Delete
    Application.DisplayAlerts = True
    On Error GoTo 0

    Dim outputSheet As Worksheet
    Set outputSheet = ActiveWorkbook.Sheets.Add(After:=ActiveWorkbook.Sheets(1))
    outputSheet.Name = "Extracted Data"
    
    ' --- Header (UPDATED: Added Rate per sq.feet) ---
    outputSheet.Range("A1:I1").Value = Array("Sr. No", "Sub Category", "Nature of Works", "Particulars", "Rate", "As per CWI (Qty)", "As per CWI (Amount)", "Rate per sq.feet", "Unit")
    outputSheet.Rows(1).Font.Bold = True
    
    Dim outputRow As Long
    outputRow = 2

    ' Step 2: Loop BOQ Sheets and Extract
    For Each targetSheet In boqSheets
    
        ' Reset Columns
        srNoCol = 0: particularsCol = 0: unitCol = 0: subCatCol = 0
        cwiHeaderRow = 0: cwiHeaderCol = 0
        cwiQtyCol = 0: cwiAmtCol = 0: rateCol = 0
        
        ' Search for Columns
        For r = 1 To 30
            For c = 1 To 100
                Dim cellVal As Variant
                cellVal = targetSheet.Cells(r, c).Value
                
                If Not IsError(cellVal) And Not IsEmpty(cellVal) Then
                    Dim val As String
                    val = Trim(LCase(CStr(cellVal)))

                    ' 1. Find Sr. No Column
                    If srNoCol = 0 And (val = "sr.no" Or val = "sr. no" Or val = "s.no" Or val = "item no" Or val = "sl.no") Then srNoCol = c
                    
                    ' 2. Find Sub Category Column
                    If subCatCol = 0 And val = "sub category as per hdfc" Then subCatCol = c

                    ' 3. Find Particulars
                    If particularsCol = 0 And (val Like "*particular*" Or val Like "*description*" Or val = "item description") Then particularsCol = c
                    
                    ' 4. CWI Header Block
                    If cwiHeaderCol = 0 And (val Like "*as per cwi*" Or val Like "*cushman*") Then
                        cwiHeaderRow = r
                        cwiHeaderCol = c
                    End If
                    
                    ' 5. Rate Column
                    If rateCol = 0 Then
                        If val = "rate" Or val = "price" Or val = "unit rate" Or val = "unit price" Or _
                           val = "current_rate" Or val = "current rate" Or _
                           val = "proposed_rate" Or val = "proposed rate" Then
                            rateCol = c
                        End If
                    End If
                    
                    ' 6. Unit
                    If unitCol = 0 And (val = "unit" Or val = "uom" Or val = "units") Then unitCol = c
                End If
            Next c
        Next r
        
        ' Fallback: If Sub Category not found, look for data keywords (Safety Net)
        If subCatCol = 0 Then
            For r = 5 To 20 
                For c = 1 To 20
                    Dim sampleTxt As String
                    sampleTxt = UCase(Trim(CStr(targetSheet.Cells(r, c).Value)))
                    If sampleTxt = "METAL-WORK" Or sampleTxt = "PAINTING" Or sampleTxt = "MISCELLANEOUS" Or sampleTxt = "FLOORING" Then
                        subCatCol = c
                        Exit For
                    End If
                Next c
                If subCatCol > 0 Then Exit For
            Next r
        End If

        If particularsCol = 0 Then GoTo NextSheet
        
        ' Find specific CWI columns (Qty/Amt)
        If cwiHeaderCol > 0 Then
            Dim searchR As Long, searchC As Long
            For searchC = cwiHeaderCol To cwiHeaderCol + 8
                For searchR = cwiHeaderRow To cwiHeaderRow + 8
                    Dim txt As String
                    txt = LCase(Trim(CStr(targetSheet.Cells(searchR, searchC).Value)))
                    If cwiQtyCol = 0 And (InStr(txt, "qty") > 0 Or InStr(txt, "quantity") > 0) Then cwiQtyCol = searchC
                    If cwiAmtCol = 0 And (InStr(txt, "amt") > 0 Or InStr(txt, "amount") > 0) Then cwiAmtCol = searchC
                Next searchR
            Next searchC
        End If
        
        startRow = cwiHeaderRow
        If startRow = 0 Then startRow = 5
        startRow = startRow + 1
        
        Dim lastRow As Long
        lastRow = targetSheet.UsedRange.Rows.Count
        
        ' Extraction Loop
        For i = startRow To lastRow
            partVal = Trim(CStr(targetSheet.Cells(i, particularsCol).Value))
            partUpper = UCase(partVal)
            
            ' 1. Skip Empty Particulars
            If partVal = "" Then GoTo NextIteration
            
            ' 2. Ignore List
            If Left(partUpper, 5) = "TOTAL" Then GoTo NextIteration
            If Left(partUpper, 11) = "GRAND TOTAL" Then GoTo NextIteration
            If InStr(partUpper, "CARRIED TO SUMMARY") > 0 Then GoTo NextIteration
            
            ' 3. Get Source Sr. No
            sourceSrNo = ""
            If srNoCol > 0 Then sourceSrNo = Trim(CStr(targetSheet.Cells(i, srNoCol).Value))
            
            ' 4. Get Source Sub Category and STANDARDIZE IT
            sourceSubCat = ""
            If subCatCol > 0 Then sourceSubCat = Trim(CStr(targetSheet.Cells(i, subCatCol).Value))
            
            ' *** STANDARDIZATION OF SUB CATEGORY ***
            sourceSubCat = StandardizeSubCategory(sourceSubCat)
            
            subCatClean = UCase(sourceSubCat)
            
            ' ---------------------------------------------------------
            ' LOGIC: MAP SUB-CATEGORY TO NATURE OF WORK
            ' ---------------------------------------------------------
            currentNatureOfWork = ""
            
            Select Case True
                ' --- CIVIL & RELATED WORKS ---
                Case InStr(subCatClean, "PLUMBING") > 0:                   currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "SANITARY") > 0:                   currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "BRICK") > 0:                      currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "PLASTER") > 0:                    currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "FLOORING") > 0:                   currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "FLOORS") > 0:                  currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "FLOOR") > 0:                  currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "DADO") > 0:                       currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "DEMOLITION") > 0:                currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "DISMANTLING") > 0:                currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "ANTI TERMITE") > 0:               currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "WATERPROOFING") > 0:              currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "R.C.C") > 0:                      currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "RCC WORKS") > 0:                      currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "CONCRETE") > 0:                   currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "MASONRY") > 0:                    currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "MISCELLANEOUS ITEMS") > 0:        currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "RAMP") > 0:                       currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "COUNTER & THRESHOLDS") > 0:       currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "TOILET ACCESSORIES") > 0:         currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "MISCELLANEOUS ITEMS") > 0:         currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "MISCELLENEOUS ITEMS") > 0:         currentNatureOfWork = "Civil & related works"
                Case InStr(subCatClean, "CONSTRUCTION AND MASONARY WORKS") > 0:         currentNatureOfWork = "Civil & related works"

                
                ' --- CARPENTRY AND INTERIOR WORKS ---
                Case InStr(subCatClean, "DOORS") > 0 And InStr(subCatClean, "TRAP") = 0: currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "WINDOWS") > 0:                    currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "PARTITION") > 0:                  currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "PANELLING") > 0 Or InStr(subCatClean, "PANELING") > 0: currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "STORAGE") > 0:                    currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "COUNTERS") > 0:                   currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "LOOSE FURNITURE") > 0:            currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "TABLES") > 0:                     currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "RAILING") > 0:                    currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "PLYWOOD BOXING") > 0:             currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "SHUTTERS") > 0:             currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                Case InStr(subCatClean, "FAÇADE") > 0 Or InStr(subCatClean, "FACADE") > 0: currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"

                ' UPDATED LINE BELOW: Handles both spellings
                Case subCatClean = "MISCELLANEOUS" Or subCatClean = "MISCELLENEOUS": currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                
                ' --- ROLLING SHUTTER AND MS WORK ---
                Case InStr(subCatClean, "METAL-WORK") > 0:                 currentNatureOfWork = "ROLLING SHUTTER AND MS WORK"
                Case InStr(subCatClean, "ROLLING SHUTTER") > 0:            currentNatureOfWork = "ROLLING SHUTTER AND MS WORK"
                Case InStr(subCatClean, "MS WORK") > 0:                    currentNatureOfWork = "ROLLING SHUTTER AND MS WORK"
                Case InStr(subCatClean, "MS WINDOW GRILL") > 0:                    currentNatureOfWork = "ROLLING SHUTTER AND MS WORK"
                
                
                ' --- PAINTING WORKS ---
                Case InStr(subCatClean, "PAINTING") > 0:                   currentNatureOfWork = "Painting works"
                
                ' --- ELECTRIFICATION AND ALLIED WORKS ---
                Case InStr(subCatClean, "ELECTRIFICATION") > 0:            currentNatureOfWork = "ELECTRIFICATION AND ALLIED WORKS"
                Case InStr(subCatClean, "ELECTRICAL") > 0:                 currentNatureOfWork = "ELECTRIFICATION AND ALLIED WORKS"
                Case InStr(subCatClean, "LIGHT") > 0:                      currentNatureOfWork = "ELECTRIFICATION AND ALLIED WORKS"
                Case InStr(subCatClean, "DG & SERVO") > 0:                 currentNatureOfWork = "ELECTRIFICATION AND ALLIED WORKS"
                Case InStr(subCatClean, "CABLES") > 0:                     currentNatureOfWork = "ELECTRIFICATION AND ALLIED WORKS"
                Case InStr(subCatClean, "EARTHING") > 0:                   currentNatureOfWork = "ELECTRIFICATION AND ALLIED WORKS"
                
                ' --- ADDITIONAL WORKS ---
                Case InStr(subCatClean, "RE-USABLE") > 0:                  currentNatureOfWork = "Additional Works"
                Case InStr(subCatClean, "VENDOR ITEMS") > 0:               currentNatureOfWork = "Additional Works"
                Case InStr(subCatClean, "ELECTRICITY & DIESEL") > 0:       currentNatureOfWork = "Additional Works"
                Case InStr(subCatClean, "ADDITIONAL") > 0:                 currentNatureOfWork = "Additional Works"
                
                ' --- POP AND FALSE CEILING WORK ---
                Case InStr(subCatClean, "POP") > 0:                        currentNatureOfWork = "POP and false ceiling work"
                Case InStr(subCatClean, "FALSE-CEILING") > 0 Or InStr(subCatClean, "FALSE CEILING") > 0: currentNatureOfWork = "POP and false ceiling work"
                Case InStr(subCatClean, "TRAP DOORS") > 0:                 currentNatureOfWork = "POP and false ceiling work"
                Case InStr(subCatClean, "POP PUNNING") > 0:                currentNatureOfWork = "POP and false ceiling work"
                Case InStr(subCatClean, "FALSE CEILING") > 0:              currentNatureOfWork = "POP and false ceiling work"
                Case InStr(subCatClean, "FALSE CELING") > 0:              currentNatureOfWork = "POP and false ceiling work"
                Case InStr(subCatClean, "FALSE-CELING") > 0:              currentNatureOfWork = "POP and false ceiling work"
                
                ' Fallback
                Case Else
                    If currentNatureOfWork = "" Then
                        If InStr(partUpper, "CIVIL") > 0 Then currentNatureOfWork = "Civil & related works"
                        If InStr(partUpper, "PLUMBING") > 0 Then currentNatureOfWork = "Civil & related works"
                        If InStr(partUpper, "CARPENTRY") > 0 Then currentNatureOfWork = "CARPENTRY AND INTERIOR WORKS"
                        If InStr(partUpper, "ELECTRICAL") > 0 Then currentNatureOfWork = "ELECTRIFICATION AND ALLIED WORKS"
                        If InStr(partUpper, "MS WORK") > 0 Then currentNatureOfWork = "ROLLING SHUTTER AND MS WORK"
                    End If
            End Select
            
            If currentNatureOfWork = "" Then currentNatureOfWork = "Others/Uncategorized"

            ' Extract Values
            Dim unitVal As String
            Dim cwiQtyVal As Double, cwiAmtVal As Double, rateVal As Double
            Dim ratePerSqFtVal As Double
            
            unitVal = SafeGetImproved(targetSheet, i, unitCol, False)
            cwiQtyVal = SafeGetImproved(targetSheet, i, cwiQtyCol, True)
            cwiAmtVal = SafeGetImproved(targetSheet, i, cwiAmtCol, True)
            rateVal = SafeGetImproved(targetSheet, i, rateCol, True)

            ' Calculate Rate Per Sq.Ft
            ratePerSqFtVal = 0
            If areaInput > 0 Then
                ratePerSqFtVal = cwiAmtVal / areaInput
            End If

            outputSheet.Cells(outputRow, 1).Value = sourceSrNo
            outputSheet.Cells(outputRow, 2).Value = sourceSubCat
            outputSheet.Cells(outputRow, 3).Value = currentNatureOfWork
            outputSheet.Cells(outputRow, 4).Value = partVal
            outputSheet.Cells(outputRow, 5).Value = rateVal
            outputSheet.Cells(outputRow, 6).Value = cwiQtyVal
            outputSheet.Cells(outputRow, 7).Value = cwiAmtVal
            
            ' NEW COLUMN: Rate per Sq.Ft
            outputSheet.Cells(outputRow, 8).Value = Round(ratePerSqFtVal, 2)
            
            ' Unit shifted to column 9
            outputSheet.Cells(outputRow, 9).Value = unitVal
            
            outputRow = outputRow + 1
            
NextIteration:
        Next i
NextSheet:
    Next targetSheet

    outputSheet.Columns.AutoFit
    ' PASSING AREA TO SUMMARY
    GenerateSmartSummary outputSheet, areaInput
    
CleanUp:
    Application.ScreenUpdating = True
    Application.Calculation = xlCalculationAutomatic
    Exit Sub

ErrorHandler:
    Resume CleanUp
End Sub

Function StandardizeSubCategory(inputText As String) As String
    Dim result As String
    Dim upperText As String
    
    result = Trim(inputText)
    If result = "" Then
        StandardizeSubCategory = ""
        Exit Function
    End If
    
    upperText = UCase(result)
    
    ' 1) R.C.C Works - Standardize all RCC variations
    If InStr(upperText, "RCC") > 0 Or InStr(upperText, "R.C.C") > 0 Then
        If InStr(upperText, "WORK") > 0 Then
            StandardizeSubCategory = "R.C.C Works"
            Exit Function
        End If
    End If
    
    ' 2) Brick and Plaster works
    If InStr(upperText, "BRICK") > 0 And InStr(upperText, "PLASTER") > 0 Then
        StandardizeSubCategory = "Brick and Plaster works"
        Exit Function
    End If
    
    ' 3) Demolition / Clearing Works
    If InStr(upperText, "DEMOLITION") > 0 And InStr(upperText, "CLEAR") > 0 Then
        StandardizeSubCategory = "Demolition / Clearing Works"
        Exit Function
    End If
    
    ' 4) Miscellaneous
    If upperText = "MISCELLANEOUS" Or upperText = "MISCELLENEOUS" Then
        StandardizeSubCategory = "Miscellaneous"
        Exit Function
    End If
    
    ' 5) Plain Cement Concrete / IPS Flooring
    If upperText = "PCC" Or InStr(upperText, "PLAIN CEMENT CONCRETE") > 0 Then
        StandardizeSubCategory = "Plain Cement Concrete / IPS Flooring"
        Exit Function
    End If
    
    ' 6) Plumbing Fittings (check this BEFORE Plumbing Works)
    If InStr(upperText, "PLUMBING FITTING") > 0 Then
        StandardizeSubCategory = "Plumbing Fittings"
        Exit Function
    End If
    
    ' 7) Plumbing (was Plumbing Works)
    If InStr(upperText, "PLUMBING") > 0 And InStr(upperText, "FITTING") = 0 Then
        StandardizeSubCategory = "Plumbing"
        Exit Function
    End If
    
    ' 8) Waterproofing treatment
    If InStr(upperText, "WATERPROOFING") > 0 And InStr(upperText, "TREATMENT") > 0 Then
        StandardizeSubCategory = "Waterproofing treatment"
        Exit Function
    End If
    
    ' 9) Anti Termite Treatment
    If InStr(upperText, "ANTI TERMITE") > 0 Then
        StandardizeSubCategory = "Anti Termite Treatment"
        Exit Function
    End If
    
    ' 10) Construction of Masonry Walls
    If InStr(upperText, "CONSTRUCTION") > 0 And InStr(upperText, "MASONRY") > 0 And InStr(upperText, "WALL") > 0 Then
        StandardizeSubCategory = "Construction of Masonry Walls"
        Exit Function
    End If
    
    ' 11) False Ceiling
    If InStr(upperText, "FALSE") > 0 And (InStr(upperText, "CEILING") > 0 Or InStr(upperText, "CELING") > 0) Then
        StandardizeSubCategory = "False Ceiling"
        Exit Function
    End If
    
    ' 12) POP Punning
    If InStr(upperText, "POP") > 0 And InStr(upperText, "PUNNING") > 0 Then
        StandardizeSubCategory = "POP Punning"
        Exit Function
    End If
    
    ' 13) Trap Doors
    If InStr(upperText, "TRAP") > 0 And InStr(upperText, "DOOR") > 0 Then
        StandardizeSubCategory = "Trap Doors"
        Exit Function
    End If
    
    ' 14) Paneling Work
    If (InStr(upperText, "PANEL") > 0 Or InStr(upperText, "PANELLING") > 0) And InStr(upperText, "WORK") > 0 Then
        StandardizeSubCategory = "Paneling Work"
        Exit Function
    End If
    
    ' 15) Partition
    If upperText = "PARTITIONS" Or upperText = "PARTITION" Then
        StandardizeSubCategory = "Partition"
        Exit Function
    End If
    
    ' 16) Storage Units and Counters
    If InStr(upperText, "STORAGE") > 0 And InStr(upperText, "COUNTER") > 0 Then
        StandardizeSubCategory = "Storage Units and Counters"
        Exit Function
    End If
    
    ' 17) Storages (single word storage)
    If upperText = "STORAGE" Or upperText = "STORAGES" Then
        StandardizeSubCategory = "Storages"
        Exit Function
    End If
    
    ' 18) PAINTING WORKS
    If InStr(upperText, "PAINTING") > 0 Then
        StandardizeSubCategory = "PAINTING WORKS"
        Exit Function
    End If
    
    ' 19) ROLLING SHUTTER AND MS WORK
    If (InStr(upperText, "ROLLING SHUTTER") > 0 Or InStr(upperText, "MS WORK") > 0) Then
        StandardizeSubCategory = "ROLLING SHUTTER AND MS WORK"
        Exit Function
    End If
    
    ' 20) ELECTRIFICATION AND ALLIED WORKS
    If InStr(upperText, "ELECTRICAL") > 0 Or InStr(upperText, "ELECTRIFICATION") > 0 Then
        StandardizeSubCategory = "ELECTRIFICATION AND ALLIED WORKS"
        Exit Function
    End If
    
    ' 21) Additional Work
    If InStr(upperText, "ADDITIONAL") > 0 And (InStr(upperText, "WORK") > 0 Or InStr(upperText, "WORKS") > 0) Then
        StandardizeSubCategory = "Additional Work"
        Exit Function
    End If
    
    ' 22) Sanitary wares ( Pure white colour )
    If InStr(upperText, "SANITARY") > 0 And InStr(upperText, "WARE") > 0 Then
        StandardizeSubCategory = "Sanitary wares ( Pure white colour )"
        Exit Function
    End If
        
    ' If no match, return original
    StandardizeSubCategory = result
End Function

Sub GenerateSmartSummary(extractedSheet As Worksheet, totalArea As Double)
    Dim summarySheet As Worksheet
    Dim lastRow As Long, i As Long
    Dim dictAmt As Object
    Dim key As Variant
    
    Dim othersAmount As Double
    Dim hasOthers As Boolean
    othersAmount = 0
    hasOthers = False
    
    Set dictAmt = CreateObject("Scripting.Dictionary")
    
    ' Default safety
    If totalArea <= 0 Then totalArea = 1
    
    Set summarySheet = ActiveWorkbook.Sheets.Add(After:=ActiveWorkbook.Sheets(extractedSheet.Index))
    summarySheet.Name = "Summary Sheet"
    
    summarySheet.Range("A1:D1").Value = Array("Sr. No", "Particulars", "Amount As per CWI", "Rate per sq.feet")
    summarySheet.Rows(1).Font.Bold = True
    
    ' Note: Extracted data still uses column 3 for nature and column 7 for Amount
    ' So this summary logic remains correct even with the new column inserted at 8
    lastRow = extractedSheet.Cells(extractedSheet.Rows.Count, 4).End(xlUp).Row
    
    For i = 2 To lastRow
        Dim nature As String
        Dim amt As Double
        
        nature = Trim(CStr(extractedSheet.Cells(i, 3).Value))
        If nature = "" Then nature = "Others/Uncategorized"
        
        amt = 0
        If IsNumeric(extractedSheet.Cells(i, 7).Value) Then
            amt = CDbl(extractedSheet.Cells(i, 7).Value)
        End If
        
        If dictAmt.Exists(nature) Then
            dictAmt(nature) = dictAmt(nature) + amt
        Else
            dictAmt.Add nature, amt
        End If
    Next i
    
    Dim outRow As Long
    Dim romIndex As Integer
    Dim grandTotalAmt As Double
    outRow = 2
    romIndex = 1
    grandTotalAmt = 0
    
    For Each key In dictAmt.Keys
        If key = "Others/Uncategorized" Then
            othersAmount = dictAmt(key)
            hasOthers = True
        Else
            Dim currAmt As Double
            currAmt = dictAmt(key)
            
            summarySheet.Cells(outRow, 1).Value = ToRoman(romIndex)
            summarySheet.Cells(outRow, 2).Value = key
            summarySheet.Cells(outRow, 3).Value = currAmt
            summarySheet.Cells(outRow, 3).NumberFormat = "#,##0.00"
            
            If totalArea > 1 Then
                summarySheet.Cells(outRow, 4).Value = Round(currAmt / totalArea, 0)
            Else
                summarySheet.Cells(outRow, 4).Value = 0
            End If
            
            grandTotalAmt = grandTotalAmt + currAmt
            outRow = outRow + 1
            romIndex = romIndex + 1
        End If
    Next key
    
    If hasOthers Then
        summarySheet.Cells(outRow, 1).Value = ToRoman(romIndex)
        summarySheet.Cells(outRow, 2).Value = "Others/Uncategorized"
        summarySheet.Cells(outRow, 3).Value = othersAmount
        summarySheet.Cells(outRow, 3).NumberFormat = "#,##0.00"
        
        If totalArea > 1 Then
            summarySheet.Cells(outRow, 4).Value = Round(othersAmount / totalArea, 0)
        Else
            summarySheet.Cells(outRow, 4).Value = 0
        End If
        grandTotalAmt = grandTotalAmt + othersAmount
        outRow = outRow + 1
    End If
    
    summarySheet.Cells(outRow, 1).Value = "Total"
    summarySheet.Cells(outRow, 1).Font.Bold = True
    summarySheet.Cells(outRow, 2).Value = "Total"
    summarySheet.Cells(outRow, 2).Font.Bold = True
    summarySheet.Cells(outRow, 3).Value = grandTotalAmt
    summarySheet.Cells(outRow, 3).Font.Bold = True
    summarySheet.Cells(outRow, 3).NumberFormat = "#,##0.00"
    
    If totalArea > 1 Then
        summarySheet.Cells(outRow, 4).Value = Round(grandTotalAmt / totalArea, 0)
        summarySheet.Cells(outRow, 4).Font.Bold = True
    End If
    
    summarySheet.Columns.AutoFit
End Sub

Function SafeGetImproved(ws As Worksheet, Row As Long, col As Long, isNumericFlag As Boolean) As Variant
    If col = 0 Then
        SafeGetImproved = IIf(isNumericFlag, 0, "")
        Exit Function
    End If
    On Error Resume Next
    Dim cellValue As Variant
    cellValue = ws.Cells(Row, col).Value
    If IsError(cellValue) Or IsEmpty(cellValue) Or Trim(CStr(cellValue)) = "" Then
        SafeGetImproved = IIf(isNumericFlag, 0, "")
    Else
        If isNumericFlag Then
            If IsNumeric(cellValue) Then
                SafeGetImproved = Round(CDbl(cellValue), 2)
            Else
                SafeGetImproved = 0
            End If
        Else
            SafeGetImproved = Trim(CStr(cellValue))
        End If
    End If
    On Error GoTo 0
End Function

Function ToRoman(n As Integer) As String
    Dim roman As String
    roman = ""
    Select Case n
        Case 1: roman = "I"
        Case 2: roman = "II"
        Case 3: roman = "III"
        Case 4: roman = "IV"
        Case 5: roman = "V"
        Case 6: roman = "VI"
        Case 7: roman = "VII"
        Case 8: roman = "VIII"
        Case 9: roman = "IX"
        Case 10: roman = "X"
        Case 11: roman = "XI"
        Case 12: roman = "XII"
        Case 13: roman = "XIII"
        Case 14: roman = "XIV"
        Case 15: roman = "XV"
        Case 16: roman = "XVI"
        Case 17: roman = "XVII"
        Case 18: roman = "XVIII"
        Case 19: roman = "XIX"
        Case 20: roman = "XX"
        Case Else: roman = CStr(n)
    End Select
    ToRoman = roman
End Function
"""

def run_excel_macro_on_folder(folder_path, macro_code):
    """
    OPTIMIZED VERSION - Creates ONE master workbook with macro,
    then applies it to all target files
    """
    if not ospath.isdir(folder_path):
        print(f"Error: Folder not found - {folder_path}")
        return

    # Initialize Excel
    try:
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        excel.ScreenUpdating = False
        excel.EnableEvents = False
        excel.AskToUpdateLinks = False
        excel.AlertBeforeOverwriting = False
    except Exception as e:
        print("Failed to initialize Excel.")
        print(e)
        return

    print(f"--- Starting Optimized Processing in: {folder_path} ---\n")

    # CREATE MASTER WORKBOOK WITH MACRO (ONCE!)
    tool_wb = None
    try:
        tool_wb = excel.Workbooks.Add()
        # Add a placeholder reference to allow code injection if needed
        excel_module = tool_wb.VBProject.VBComponents.Add(1)
        excel_module.CodeModule.AddFromString(macro_code)
        
        macro_name = f"'{tool_wb.Name}'!ExtractCWI_Data"
        print(f"✓ Master workbook created: {tool_wb.Name}")
        print(f"✓ Macro compiled and ready\n")
        print("=" * 70)

    except Exception as e:
        print("❌ Error creating Master Workbook.")
        print("   Ensure 'Trust Access to VBA Project' is enabled in Excel.")
        print(f"   Error: {e}")
        excel.Quit()
        return

    # Get list of files to process
    files_to_process = []
    try:
        import os
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.xlsx', '.xlsm', '.xls')) and not filename.startswith('~$'):
                files_to_process.append(filename)
    except Exception as e:
        print(f"Error listing files: {e}")
        excel.Quit()
        return
    
    total_files = len(files_to_process)
    print(f"Found {total_files} Excel files to process\n")
    print("=" * 70)

    # Process each file
    count_success = 0
    count_failed = 0
    
    for idx, filename in enumerate(files_to_process, 1):
        start_time = time.time()
        
        # Extract area from filename
        match = re.search(r"^(\d+)\s*sqft", filename, re.IGNORECASE)
        area_found = float(match.group(1)) if match else 1.0
        
        file_path = os.path.join(folder_path, filename)
        
        print(f"[{idx}/{total_files}] Processing: {filename}")
        print(f"           Area: {area_found} sqft")
        
        target_wb = None
        try:
            # Open target workbook
            target_wb = excel.Workbooks.Open(file_path, UpdateLinks=False, ReadOnly=False)
            
            # Run macro from master workbook on target
            excel.Run(macro_name, area_found)
            
            # Save and close
            target_wb.Save()
            target_wb.Close(SaveChanges=False)
            
            elapsed = time.time() - start_time
            print(f"           ✓ SUCCESS in {elapsed:.1f}s")
            count_success += 1
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"           ❌ FAILED in {elapsed:.1f}s")
            print(f"           Error: {str(e)[:100]}")
            count_failed += 1
            
            # Try to close without saving
            try:
                if target_wb is not None:
                    target_wb.Close(SaveChanges=False)
            except:
                pass
        
        print("-" * 70)

    # Cleanup
    print("\nCleaning up...")
    try:
        if tool_wb is not None:
            tool_wb.Close(SaveChanges=False)
    except:
        pass
    
    excel.ScreenUpdating = True
    excel.EnableEvents = True
    excel.Quit()
    
    # Final summary
    print("=" * 70)
    print(f"PROCESSING COMPLETE!")
    print(f"  ✓ Successful: {count_success}")
    print(f"  ❌ Failed: {count_failed}")
    print(f"  Total: {total_files}")
    print("=" * 70)

if __name__ == "__main__":
    run_excel_macro_on_folder(FOLDER_PATH, VBA_MACRO_OPTIMIZED)