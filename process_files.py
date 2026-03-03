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
FOLDER_PATH = r"C:\Users\Tanush.Bidkar\Downloads\richard"

# --- OPTIMIZED VBA TEMPLATE (Updated with Rate per Sq.Ft) ---
VBA_MACRO_OPTIMIZED = """
Option Explicit

'GLOBAL VARIABLE TO STORE AREA (Passed from Python)
Public G_TotalArea As Double

'#########################################################################################
'# MAIN MACRO: ENTRY POINT
'# Accepts Area from Python, sets the global variable, then runs logic on ACTIVE WORKBOOK
'#########################################################################################
Sub ExtractCWI_Data_Final_TypoFix(areaInput As Double)
    
    ' Set the global area variable
    If areaInput = 0 Then G_TotalArea = 1 Else G_TotalArea = areaInput

    '--- Settings ---
    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual
    Application.EnableEvents = False
    On Error GoTo ErrorHandler

    '--- Variables ---
    Dim targetSheet As Worksheet, ws As Worksheet
    Dim uCodeCol As Long, particularsCol As Long, unitCol As Long
    Dim cwiQtyCol As Long, cwiAmtCol As Long, rateCol As Long
    Dim cwiHeaderRow As Long, cwiHeaderCol As Long
    Dim startRow As Long, r As Long, c As Long, i As Long
    
    '--- Variables for Nature Tracking ---
    Dim derivedSubCat As String, derivedNature As String
    Dim uCodeRaw As String, uCodeClean As String
    Dim partTextClean As String
    
    '--- MEMORY VARIABLES ---
    Dim lastValidSubCat As String
    Dim lastValidNature As String
    
    '--- State Machine for "Additional Works" Zone ---
    Dim InAdditionalWorksZone As Boolean
    Dim AdditionalWorksStartRoman As String
    Dim AdditionalWorksStopRoman As String
    
    '--- Collection for BOQ sheets ---
    Dim boqSheets As New Collection
    Dim processedSheetCount As Long
    processedSheetCount = 0
    
    '--- Sheet Detection (EXPANDED LIST) ---
    For Each ws In ActiveWorkbook.Worksheets
        Dim wsNameLower As String
        wsNameLower = LCase(ws.Name)
        
        ' 1. Ignore Summary/Extracted Data sheets
        If Not (wsNameLower Like "*summary*" Or wsNameLower = "extracted data") Then
            
            ' 2. EXPANDED CHECK - Added "Consolidated BOQ" and "Consolidated Bill"
            If wsNameLower Like "boq*" Or _
               wsNameLower Like "bill*" Or _
               wsNameLower Like "*consolidated boq*" Or _
               wsNameLower Like "*consolidated bill*" Or _
               wsNameLower Like "*work done*" Or _
               wsNameLower Like "*measurement*" Or _
               wsNameLower Like "*cwi*" Or _
               wsNameLower Like "*post audit*" Or _
               wsNameLower Like "*abstract*" Then
               
               boqSheets.Add ws
            End If
            
        End If
    Next ws

    If boqSheets.Count = 0 Then GoTo CleanUp

    ' Step 1: Create/Reset "Extracted Data" Sheet
    On Error Resume Next
    Application.DisplayAlerts = False
    ActiveWorkbook.Sheets("Extracted Data").Delete
    ActiveWorkbook.Sheets("Summary Sheet").Delete
    Application.DisplayAlerts = True
    On Error GoTo ErrorHandler

    Dim outputSheet As Worksheet
    Set outputSheet = ActiveWorkbook.Sheets.Add(After:=ActiveWorkbook.Sheets(1))
    outputSheet.Name = "Extracted Data"
    
    ' --- Header (UPDATED with Rate per sq.feet) ---
    outputSheet.Range("A1:I1").Value = Array("U. Code", "Sub Category", "Nature of Works", "Particulars", "Rate", "As per CWI (Qty)", "As per CWI (Amount)", "Rate per sq.feet", "Unit")
    outputSheet.Rows(1).Font.Bold = True
    
    Dim outputRow As Long
    outputRow = 2

    ' Step 2: Loop BOQ Sheets and Extract
    For Each targetSheet In boqSheets
    
        ' Reset Columns & Flags per sheet
        uCodeCol = 0: particularsCol = 0: unitCol = 0
        cwiHeaderRow = 0: cwiHeaderCol = 0
        cwiQtyCol = 0: cwiAmtCol = 0: rateCol = 0
        InAdditionalWorksZone = False
        AdditionalWorksStartRoman = ""
        AdditionalWorksStopRoman = ""
        
        ' Reset Memory for new sheet
        lastValidSubCat = ""
        lastValidNature = ""
        
        ' Search for Headers - EXPANDED TO 50 ROWS (was 30)
        For r = 1 To 50
            For c = 1 To 100
                Dim cellVal As Variant
                cellVal = targetSheet.Cells(r, c).Value
                
                If Not IsError(cellVal) And Not IsEmpty(cellVal) Then
                    Dim val As String
                    val = Trim(LCase(CStr(cellVal)))

                    ' EXPANDED U.CODE/SR.NO DETECTION - Scan first 10 rows regardless of other content
                    If r <= 10 Then
                        If uCodeCol = 0 And (val = "u.code" Or val = "u. code" Or val = "u code" Or _
                           val = "code" Or val = "item code" Or val Like "sr*no*" Or _
                           val = "sr no" Or val = "sr. no" Or val = "sr.no" Or val = "s.no" Or _
                           val = "serial no" Or val = "item no") Then
                            uCodeCol = c
                        End If
                    End If
                    
                    ' PARTICULARS - Can be found anywhere in search range
                    If particularsCol = 0 And (val Like "*particular*" Or val Like "*description*" Or val = "item description") Then 
                        particularsCol = c
                    End If
                    
                    ' CWI HEADER
                    If cwiHeaderCol = 0 And (val Like "*as per cwi*" Or val Like "*cushman*") Then
                        cwiHeaderRow = r
                        cwiHeaderCol = c
                    End If
                    
                    ' --- UPDATED: ROBUST RATE/PRICE DETECTION ---
                    If rateCol = 0 Then
                        If val = "rate" Or val = "price" Or val = "unit rate" Or val = "unit price" Or _
                           val = "current_rate" Or val = "current rate" Or _
                           val = "proposed_rate" Or val = "proposed rate" Then 
                            rateCol = c
                        End If
                    End If
                    
                    ' UNIT
                    If unitCol = 0 And (val = "unit" Or val = "uom" Or val = "units") Then 
                        unitCol = c
                    End If
                End If
            Next c
        Next r
        
        If uCodeCol = 0 Or particularsCol = 0 Then GoTo NextSheet
        
        ' Find specific CWI columns
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
        
        ' Determine start row - use header row or default
        startRow = cwiHeaderRow
        If startRow = 0 Then
            ' If no CWI header found, start after U.Code header
            If uCodeCol > 0 Then
                ' Find the row where U.Code header is
                For r = 1 To 50
                    Dim checkVal As String
                    checkVal = LCase(Trim(CStr(targetSheet.Cells(r, uCodeCol).Value)))
                    If checkVal = "u.code" Or checkVal = "u. code" Or checkVal = "code" Or _
                       checkVal Like "sr*no*" Or checkVal = "sr no" Or checkVal = "sr. no" Then
                        startRow = r
                        Exit For
                    End If
                Next r
            End If
            
            ' Final fallback
            If startRow = 0 Then startRow = 10
        End If
        
        startRow = startRow + 1
        
        Dim lastRow As Long
        lastRow = targetSheet.UsedRange.Rows.Count
        
        ' Extraction Loop
        For i = startRow To lastRow
            Dim partVal As String
            partVal = Trim(CStr(targetSheet.Cells(i, particularsCol).Value))
            partTextClean = UCase(partVal)
            
            uCodeRaw = ""
            If uCodeCol > 0 Then uCodeRaw = Trim(CStr(targetSheet.Cells(i, uCodeCol).Value))
            uCodeClean = UCase(uCodeRaw)
            
            If partVal = "" Then GoTo NextIteration
            If ShouldIgnoreRow(partTextClean, uCodeClean) Then GoTo NextIteration

            ' Fix Typo
            If uCodeClean = "VII_PART-3" Then uCodeClean = "VI_PART-3"

            ' NEW DYNAMIC ADDITIONAL WORKS ZONE LOGIC
            If Not InAdditionalWorksZone Then
                ' Check if "ADDITIONAL" keyword appears in particulars AND u.code is a new roman numeral
                If (InStr(partTextClean, "ADDITIONAL WORK") > 0 Or InStr(partTextClean, "ADDITIONAL CHARGE") > 0) Then
                    Dim romanFound As String
                    romanFound = ExtractRomanNumeral(uCodeClean)
                    
                    If romanFound <> "" Then
                        InAdditionalWorksZone = True
                        AdditionalWorksStartRoman = romanFound
                        AdditionalWorksStopRoman = GetNextRoman(romanFound)
                    End If
                End If
            End If
            
            ' Check if we should EXIT Additional Works Zone
            If InAdditionalWorksZone Then
                ' Exit if we hit the next roman numeral (the stop roman)
                Dim currentRoman As String
                currentRoman = ExtractRomanNumeral(uCodeClean)
                
                If currentRoman <> "" And currentRoman <> AdditionalWorksStartRoman Then
                    If IsRomanEqualOrGreater(currentRoman, AdditionalWorksStopRoman) Then
                        InAdditionalWorksZone = False
                    End If
                End If
                
                ' Also exit if ACCESSIBILITY keyword found
                If InStr(partTextClean, "ACCESSIBILITY") > 0 Then
                    InAdditionalWorksZone = False
                End If
            End If
            
            ' Mapping Logic
            derivedSubCat = "" : derivedNature = ""
            
            If InAdditionalWorksZone Then
                derivedSubCat = "Additional Work"
                derivedNature = "Additional Work"
                lastValidSubCat = derivedSubCat
                lastValidNature = derivedNature
            Else
                Call GetCategoryAndNature(uCodeClean, partTextClean, derivedSubCat, derivedNature)
                If derivedNature <> "" Then
                    lastValidSubCat = derivedSubCat
                    lastValidNature = derivedNature
                Else
                    If uCodeClean <> "" And lastValidNature <> "" Then
                        derivedSubCat = lastValidSubCat
                        derivedNature = lastValidNature
                    Else
                        If uCodeClean <> "" Then
                             derivedSubCat = "Others"
                             derivedNature = "Others/Uncategorized"
                        End If
                    End If
                End If
            End If

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
            If G_TotalArea > 0 Then
                ratePerSqFtVal = cwiAmtVal / G_TotalArea
            End If

            outputSheet.Cells(outputRow, 1).Value = uCodeRaw
            outputSheet.Cells(outputRow, 2).Value = derivedSubCat
            outputSheet.Cells(outputRow, 3).Value = derivedNature
            outputSheet.Cells(outputRow, 4).Value = partVal
            outputSheet.Cells(outputRow, 5).Value = rateVal
            outputSheet.Cells(outputRow, 6).Value = cwiQtyVal
            outputSheet.Cells(outputRow, 7).Value = cwiAmtVal
            
            ' NEW COLUMN: Rate per sq.feet
            outputSheet.Cells(outputRow, 8).Value = Round(ratePerSqFtVal, 2)
            
            ' Shifted Unit to Col 9
            outputSheet.Cells(outputRow, 9).Value = unitVal
            
            outputRow = outputRow + 1
            
NextIteration:
        Next i
        processedSheetCount = processedSheetCount + 1
NextSheet:
    Next targetSheet

    outputSheet.Columns.AutoFit
    
    ' Step 3: Generate the Summary Sheet
    GenerateSmartSummary outputSheet
    
CleanUp:
    Application.ScreenUpdating = True
    Application.Calculation = xlCalculationAutomatic
    Application.EnableEvents = True
    Exit Sub

ErrorHandler:
    Application.ScreenUpdating = True
    Application.Calculation = xlCalculationAutomatic
    Application.EnableEvents = True
    MsgBox "Error: " & Err.Description
End Sub

'#########################################################################################
'# HELPER FUNCTION: Extract Roman Numeral from U.Code
'# Returns the pure roman numeral (VII, VIII, IX, X, etc.) if found at start, else ""
'#########################################################################################
Function ExtractRomanNumeral(uCode As String) As String
    ExtractRomanNumeral = ""
    If uCode = "" Then Exit Function
    
    Dim romanNumerals As Variant
    romanNumerals = Array("VIII", "VII", "IX", "VI", "X", "V", "XI", "IV", "XII", "III", "XIII", "II", "XIV", "XV", "I")
    
    Dim i As Integer
    For i = 0 To UBound(romanNumerals)
        Dim roman As String
        roman = CStr(romanNumerals(i))
        
        ' Check if uCode starts with this roman numeral followed by delimiter or end
        If uCode = roman Then
            ExtractRomanNumeral = roman
            Exit Function
        End If
        
        If Left(uCode, Len(roman) + 1) = roman & "_" Or _
           Left(uCode, Len(roman) + 1) = roman & "." Or _
           Left(uCode, Len(roman) + 1) = roman & "-" Then
            ExtractRomanNumeral = roman
            Exit Function
        End If
    Next i
End Function

'#########################################################################################
'# HELPER FUNCTION: Get Next Roman Numeral
'# Returns the next roman in sequence (VII -> VIII, VIII -> IX, etc.)
'#########################################################################################
Function GetNextRoman(currentRoman As String) As String
    Select Case currentRoman
        Case "I": GetNextRoman = "II"
        Case "II": GetNextRoman = "III"
        Case "III": GetNextRoman = "IV"
        Case "IV": GetNextRoman = "V"
        Case "V": GetNextRoman = "VI"
        Case "VI": GetNextRoman = "VII"
        Case "VII": GetNextRoman = "VIII"
        Case "VIII": GetNextRoman = "IX"
        Case "IX": GetNextRoman = "X"
        Case "X": GetNextRoman = "XI"
        Case "XI": GetNextRoman = "XII"
        Case "XII": GetNextRoman = "XIII"
        Case "XIII": GetNextRoman = "XIV"
        Case "XIV": GetNextRoman = "XV"
        Case "XV": GetNextRoman = "XVI"
        Case Else: GetNextRoman = "XX"
    End Select
End Function

'#########################################################################################
'# HELPER FUNCTION: Compare Roman Numerals
'# Returns True if roman1 >= roman2
'#########################################################################################
Function IsRomanEqualOrGreater(roman1 As String, roman2 As String) As Boolean
    IsRomanEqualOrGreater = (RomanToInt(roman1) >= RomanToInt(roman2))
End Function

Function RomanToInt(roman As String) As Integer
    Select Case roman
        Case "I": RomanToInt = 1
        Case "II": RomanToInt = 2
        Case "III": RomanToInt = 3
        Case "IV": RomanToInt = 4
        Case "V": RomanToInt = 5
        Case "VI": RomanToInt = 6
        Case "VII": RomanToInt = 7
        Case "VIII": RomanToInt = 8
        Case "IX": RomanToInt = 9
        Case "X": RomanToInt = 10
        Case "XI": RomanToInt = 11
        Case "XII": RomanToInt = 12
        Case "XIII": RomanToInt = 13
        Case "XIV": RomanToInt = 14
        Case "XV": RomanToInt = 15
        Case Else: RomanToInt = 99
    End Select
End Function

'#########################################################################################
'# CORE LOGIC: MAP U.CODE TO SUB CATEGORY & NATURE
'#########################################################################################
Sub GetCategoryAndNature(uCode As String, partText As String, ByRef subCat As String, ByRef nature As String)
    Dim uc As String
    uc = Replace(uCode, " ", "")
    subCat = "" : nature = ""
    
    ' I: CIVIL
    If CheckPrefix(uc, "I_1") Then subCat = "Demolition / Clearing Works": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_2") Then subCat = "Construction of Masonry Walls": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_3") Then subCat = "R.C.C Works": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_4") Then subCat = "Plaster": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_5") Then subCat = "Waterproofing treatment": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_6") Then subCat = "Brick Bat Coba": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_7") Then subCat = "Plain Cement Concrete / IPS Flooring": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_8") Then subCat = "Drywall Panel": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_9") Then subCat = "Flooring": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_10") Then subCat = "Pantry / Dining / Toilet Counter & Thresholds": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_11") Then subCat = "External Paving/Flooring/Window sill": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_12") Then subCat = "Facade Treatment": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_13") Then subCat = "Handicapped Access Ramp": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_14") Then subCat = "Ramp Finishes": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_15") Then subCat = "Plumbing": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_16") Then subCat = "Sanitary wares ( Pure white colour )": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_17") Then subCat = "Plumbing Fittings": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_18") Then subCat = "Toilet Accessories": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_19") Then subCat = "External plumbing works.": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_20") Then subCat = "Anti Termite Treatment": nature = "CIVIL & RELATED WORKS": Exit Sub
    If CheckPrefix(uc, "I_21") Then subCat = "Miscellaneous Items": nature = "CIVIL & RELATED WORKS": Exit Sub
    
    ' II: POP
    If uc = "II" Or CheckPrefix(uc, "II_1") Then subCat = "POP Punning": nature = "POP & FALSE CEILING WORKS": Exit Sub
    If CheckPrefix(uc, "II_2") Then subCat = "False Ceiling": nature = "POP & FALSE CEILING WORKS": Exit Sub
    If CheckPrefix(uc, "II_3") Then subCat = "Trap Doors": nature = "POP & FALSE CEILING WORKS": Exit Sub
    
    ' III: CARPENTRY
    If uc = "III" Or CheckPrefix(uc, "III_1") Then subCat = "Partition": nature = "CARPENTRY AND INTERIOR WORKS": Exit Sub
    If CheckPrefix(uc, "III_2") Then subCat = "Paneling Work": nature = "CARPENTRY AND INTERIOR WORKS": Exit Sub
    If CheckPrefix(uc, "III_3") Then subCat = "Plywood Boxing": nature = "CARPENTRY AND INTERIOR WORKS": Exit Sub
    If CheckPrefix(uc, "III_4") Then subCat = "Doors": nature = "CARPENTRY AND INTERIOR WORKS": Exit Sub
    If CheckPrefix(uc, "III_5") Or CheckPrefix(uc, "III_6") Then subCat = "Tables": nature = "CARPENTRY AND INTERIOR WORKS": Exit Sub
    If CheckPrefix(uc, "III_7") Then subCat = "Shutters": nature = "CARPENTRY AND INTERIOR WORKS": Exit Sub
    If CheckPrefix(uc, "III_8") Then subCat = "Façade Treatment": nature = "CARPENTRY AND INTERIOR WORKS": Exit Sub
    If CheckPrefix(uc, "III_9") Or CheckPrefix(uc, "III_10") Then subCat = "Railing": nature = "CARPENTRY AND INTERIOR WORKS": Exit Sub
    If CheckPrefix(uc, "III_11") Then subCat = "Windows": nature = "CARPENTRY AND INTERIOR WORKS": Exit Sub
    If CheckPrefix(uc, "III_12") Then subCat = "Miscellaneous": nature = "CARPENTRY AND INTERIOR WORKS": Exit Sub
    
    ' IV - VI
    If Left(uc, 2) = "IV" Then subCat = "PAINTING WORKS": nature = "PAINTING WORKS": Exit Sub
    If Left(uc, 1) = "V" And Left(uc, 2) <> "VI" And Left(uc, 3) <> "VII" Then subCat = "ROLLING SHUTTER AND MS WORK": nature = "ROLLING SHUTTER AND MS WORK": Exit Sub
    If Left(uc, 2) = "VI" And Left(uc, 3) <> "VII" Then subCat = "ELECTRIFICATION AND ALLIED WORKS": nature = "ELECTRIFICATION AND ALLIED WORKS": Exit Sub
End Sub

Function CheckPrefix(code As String, prefix As String) As Boolean
    Dim pLen As Integer
    pLen = Len(prefix)
    If code = prefix Then CheckPrefix = True: Exit Function
    If Left(code, pLen + 1) = prefix & "." Or Left(code, pLen + 1) = prefix & "_" Then CheckPrefix = True: Exit Function
    CheckPrefix = False
End Function

Function ShouldIgnoreRow(partTxt As String, uCode As String) As Boolean
    ShouldIgnoreRow = False
    If uCode = "" And Left(partTxt, 5) <> "TOTAL" Then ShouldIgnoreRow = False: Exit Function
    If uCode = "TOTAL" Then ShouldIgnoreRow = True: Exit Function
    If Left(partTxt, 5) = "TOTAL" Then ShouldIgnoreRow = True: Exit Function
    If Left(partTxt, 11) = "GRAND TOTAL" Then ShouldIgnoreRow = True: Exit Function
    If InStr(partTxt, "CARRIED TO SUMMARY") > 0 Then ShouldIgnoreRow = True: Exit Function
End Function

'#########################################################################################
'# SUMMARY GENERATOR
'#########################################################################################
Sub GenerateSmartSummary(extractedSheet As Worksheet)
    Dim summarySheet As Worksheet
    Dim lastRow As Long, i As Long
    Dim dictAmt As Object
    Dim key As Variant
    Dim totalArea As Double
    
    Set dictAmt = CreateObject("Scripting.Dictionary")
    
    ' --- USE GLOBAL VARIABLE ---
    totalArea = G_TotalArea
    
    Set summarySheet = ActiveWorkbook.Sheets.Add(After:=ActiveWorkbook.Sheets(extractedSheet.Index))
    summarySheet.Name = "Summary Sheet"
    
    summarySheet.Range("A1:D1").Value = Array("Sr. No", "Particulars", "Amount As per CWI", "Rate per sq.feet")
    summarySheet.Rows(1).Font.Bold = True
    
    lastRow = extractedSheet.Cells(extractedSheet.Rows.Count, 3).End(xlUp).Row
    
    ' Loop Extracted Data
    For i = 2 To lastRow
        Dim uCode As String, subCat As String, nature As String
        Dim amt As Double
        
        uCode = Trim(CStr(extractedSheet.Cells(i, 1).Value))
        subCat = Trim(CStr(extractedSheet.Cells(i, 2).Value))
        nature = Trim(CStr(extractedSheet.Cells(i, 3).Value))
        
        amt = 0
        If IsNumeric(extractedSheet.Cells(i, 7).Value) Then amt = CDbl(extractedSheet.Cells(i, 7).Value)
        
        ' --- STRICT FILTER LOGIC ---
        Dim isValid As Boolean
        isValid = True
        
        If amt = 0 Then isValid = False
        
        If nature = "Additional Work" Then
            ' Exception
        Else
            If uCode = "" Then isValid = False
            If subCat = "" Or subCat = "Others" Then isValid = False
            If nature = "" Or nature = "Others/Uncategorized" Then isValid = False
        End If
        
        If isValid Then
            If dictAmt.Exists(nature) Then
                dictAmt(nature) = dictAmt(nature) + amt
            Else
                dictAmt.Add nature, amt
            End If
        End If
    Next i
    
    ' Output Aggregated Data
    Dim outRow As Long, romIndex As Integer
    Dim grandTotalAmt As Double
    
    outRow = 2
    romIndex = 1
    grandTotalAmt = 0
    
    For Each key in dictAmt.Keys
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
    Next key
    
    ' Grand Total
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
    If col = 0 Then SafeGetImproved = IIf(isNumericFlag, 0, ""): Exit Function
    On Error Resume Next
    Dim cellValue As Variant: cellValue = ws.Cells(Row, col).Value
    If IsError(cellValue) Or IsEmpty(cellValue) Or Trim(CStr(cellValue)) = "" Then
        SafeGetImproved = IIf(isNumericFlag, 0, "")
    Else
        If isNumericFlag Then
            If IsNumeric(cellValue) Then SafeGetImproved = Round(CDbl(cellValue), 2) Else SafeGetImproved = 0
        Else
            SafeGetImproved = Trim(CStr(cellValue))
        End If
    End If
    On Error GoTo 0
End Function

Function ToRoman(n As Integer) As String
    Dim romans As Variant
    romans = Array("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX")
    If n > 0 And n <= 20 Then ToRoman = romans(n) Else ToRoman = CStr(n)
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
        excel_module = tool_wb.VBProject.VBComponents.Add(1)
        excel_module.CodeModule.AddFromString(macro_code)
        
        macro_name = f"'{tool_wb.Name}'!ExtractCWI_Data_Final_TypoFix"
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
        
        import os
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