# Class 2: Handling Data with Python Data Structures (+ CSV I/O)

### Overview

If Class 1 taught Python to crunch numbers, Class 2 teaches it to **organize** information. Lists, tuples, sets, and dictionaries are like lab drawers: each holds different kinds of parts, and using the right drawer makes your experiments smoother. We’ll also read a simple **CSV** (voltage vs. time), compute basic stats, and visualize a tiny plot.

---

### Lecture Notes (quick, teachable bites)

* **List**: ordered, mutable. Use for sequences of readings.
* **Tuple**: ordered, immutable. Use for fixed-size records (e.g., `(time, voltage)`).
* **Set**: unique elements only. Use for deduplicating sensor IDs or tags.
* **Dict**: key→value map. Use for labeled data like `{"t": 0.1, "V": 3.3}` or config.
* **File I/O**: `open()`, `read()`, `write()`, and the `csv` module to parse files.

**Mini-analogy**

* List = a timeline of sensor readings
* Tuple = a single reading snapshot `(t, V)`
* Set = the unique types of sensors present
* Dict = a labeled measurement or config

---

## Demo Notebook (Google Colab style cells)

### 1) Lists, Tuples, Sets, Dicts — EE-flavored

```python
# Lists: ordered, mutable
voltages = [3.1, 3.3, 3.2, 3.4, 3.3]
times = [0, 1, 2, 3, 4]  # seconds
print("First voltage:", voltages[0])
voltages.append(3.5)
print("After append:", voltages)

# Tuples: ordered, immutable (good for records)
reading = (times[0], voltages[0])  # (time, voltage)
print("A single reading tuple:", reading)

# Sets: unique collection (deduplicate)
sensor_ids = ["A1", "A2", "A2", "A3", "A1"]
unique_sensors = set(sensor_ids)
print("Unique sensors:", unique_sensors)

# Dicts: key-value mapping (label your data)
config = {"sample_rate_hz": 100, "sensor": "A1", "location": "Bench-1"}
print("Config sample rate:", config["sample_rate_hz"])

# List of dicts: structured rows
rows = [
    {"t": 0, "V": 3.1},
    {"t": 1, "V": 3.3},
    {"t": 2, "V": 3.2},
]
print("Row 2 voltage:", rows[1]["V"])
```

### 2) Reading a CSV (voltage vs time) with the csv module

```python
# We'll create a small CSV file, then read it.
import csv

# Create a demo CSV in runtime (Colab friendly)
with open("voltage_time.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "voltage_v"])
    writer.writerow([0, 3.10])
    writer.writerow([1, 3.30])
    writer.writerow([2, 3.20])
    writer.writerow([3, 3.40])
    writer.writerow([4, 3.30])

# Read the CSV
times, volts = [], []
with open("voltage_time.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        times.append(float(row["time_s"]))
        volts.append(float(row["voltage_v"]))

print("Times:", times)
print("Voltages:", volts)
```

### 3) Simple stats + tiny plot

```python
# Basic statistics
avg_v = sum(volts) / len(volts)
v_min = min(volts)
v_max = max(volts)
print(f"Avg V = {avg_v:.3f}  |  Min V = {v_min:.3f}  |  Max V = {v_max:.3f}")

# Quick line plot (preview; full viz comes in Class 4)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(times, volts, marker="o")
plt.title("Voltage vs Time (Demo)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.show()
```

---

## Classworks (5) — Skeleton Code (students fill the blanks)

### Classwork 1: List Ops + Indexing

```python
# =========================================
# CLASSWORK 1: LIST OPERATIONS + INDEXING
# =========================================
# Task:
# 1) Create a list named 'readings' with 5 voltage values (floats).
# 2) Print the first and last elements by index.
# 3) Append a new value and print the updated list.
# 4) Replace the 3rd element with a new value.

# readings = [???, ???, ???, ???, ???]
# print("First element:", ???)
# print("Last element:", ???)
# readings.???(???)  # append a new value
# readings[??] = ??? # replace the 3rd element (index 2)
# print("Updated readings:", readings)
```

### Classwork 2: Tuple + Dict Records

```python
# =========================================
# CLASSWORK 2: TUPLE + DICT RECORDS
# =========================================
# Task:
# 1) Create a tuple called 'sample' representing one reading: (time_s, voltage_v).
# 2) Create a dict called 'row' with keys 't' and 'V' from that tuple.
# 3) Print them both, showing how to access the voltage value in each.

# sample = (???, ???)  # e.g., (0.5, 3.25)
# row = {"t": ???, "V": ???}
# print("Tuple voltage:", sample[??])
# print("Dict voltage:", row[???])
```

### Classwork 3: Sets for Uniqueness

```python
# =========================================
# CLASSWORK 3: SETS FOR UNIQUENESS
# =========================================
# Task:
# Given a list of sensor IDs with duplicates, create a set to find unique IDs.
# Then convert the set back to a sorted list.

# sensors = ["S1", "S2", "S1", "S3", "S2", "S4", "S3"]
# unique_set = set(???)          # fill in
# unique_sorted = sorted(list(???))  # fill in
# print("Unique sorted sensors:", unique_sorted)
```

### Classwork 4: CSV Read + Basic Stats

```python
# =========================================
# CLASSWORK 4: CSV READ + BASIC STATS
# =========================================
# Task:
# 1) Create (or reuse) a CSV file named 'vt_small.csv' with columns: time_s, voltage_v.
#    Add at least 5 rows.
# 2) Read it with csv.DictReader and store values in lists 't' and 'v'.
# 3) Compute average voltage and print it.

import csv

# Step 1: Create the CSV file
# with open("vt_small.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["time_s", "voltage_v"])
#     # Write at least 5 rows of (time, voltage)
#     # writer.writerow([???, ???])

# Step 2: Read the CSV into lists
# t, v = [], []
# with open("vt_small.csv", "r") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         t.append(???)  # convert to float
#         v.append(???)  # convert to float

# Step 3: Compute average voltage
# avg_v = ??? / ???  # sum(v) / len(v)
# print("Average voltage:", round(avg_v, 3))
```

### Classwork 5: Dict of Lists vs List of Dicts

```python
# =====================================================
# CLASSWORK 5: DICT OF LISTS vs LIST OF DICTS (CHOICE)
# =====================================================
# Task:
# You will build the same dataset in two structures and access a value from each.
# A) Dict of Lists: {"t": [...], "V": [...]}
# B) List of Dicts: [{"t": t0, "V": v0}, {"t": t1, "V": v1}, ...]

# A) Dict of Lists
# data_cols = {"t": [0, 1, 2], "V": [3.10, 3.25, 3.20]}
# # Print the voltage at index 1 from this structure:
# print("A) V at index 1:", ???)

# B) List of Dicts
# data_rows = [{"t": 0, "V": 3.10}, {"t": 1, "V": 3.25}, {"t": 2, "V": 3.20}]
# # Print the voltage from the second row:
# print("B) V from row 2:", ???)
```

---

### Wrap-up / Homework Challenge

* Extend **Classwork 4** to also compute **min** and **max** voltage, and print a small summary line:
  `Summary: n=<count>, mean=<...>, min=<...>, max=<...>`
* Bonus: Save your summary to a new text file `summary.txt` using `open("summary.txt", "w")`.

---

