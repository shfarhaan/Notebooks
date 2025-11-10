# Class 1: Python Basics

### Overview

We’ll learn Python fundamentals to simulate simple electrical circuits. Think of Python as your “digital multimeter”: it measures, calculates, and automates your experiments.

---

### Lecture Notes

Covered today:

* Variables and data types (int, float, str, bool)
* Conditionals (`if/else`)
* Loops (`for`, `while`)
* Functions (`def`)
* Circuit analogy: series vs. parallel resistors

---

### Demo Notebook Code (Instructor-led)

```python
# Class 1 Demo: Python Basics

# 1. Variables
resistor1 = 100  # ohms
resistor2 = 200  # ohms
voltage = 9      # volts
print("Resistors:", resistor1, resistor2)

# 2. Series Circuit
total_series = resistor1 + resistor2
print("Series resistance:", total_series, "ohms")

# 3. Parallel Circuit
total_parallel = 1 / ((1/resistor1) + (1/resistor2))
print("Parallel resistance:", total_parallel, "ohms")

# 4. Conditionals
if total_series > 250:
    print("Warning: resistance too high!")
else:
    print("Safe resistance range")

# 5. Loop: simulate voltage drops
for r in [resistor1, resistor2]:
    drop = voltage * (r / total_series)
    print(f"Voltage drop across {r}Ω: {round(drop, 2)} V")

# 6. Function
def equivalent_parallel(resistors):
    return 1 / sum(1/r for r in resistors)

print("Parallel eq. of [100,200,300]:", equivalent_parallel([100,200,300]), "ohms")
```

---

### Classworks (5) — Skeleton Code

```python
# ==============================
# CLASSWORK 1: Variable Practice
# ==============================
# Task: Create variables for three resistors (10Ω, 20Ω, 30Ω)
# and print them with labels.

# Example:
# resistor1 = ???
# resistor2 = ???
# resistor3 = ???

# print("Resistor 1:", ???)
# print("Resistor 2:", ???)
# print("Resistor 3:", ???)
```

```python
# ==============================
# CLASSWORK 2: Series Resistance
# ==============================
# Task: Compute the total resistance in series
# Formula: R_total = R1 + R2 + R3

# series_total = ???
# print("Total Series Resistance:", ???)
```

```python
# ==============================
# CLASSWORK 3: Parallel Resistance
# ==============================
# Task: Compute total resistance in parallel
# Formula: 1/R_total = 1/R1 + 1/R2 + 1/R3

# parallel_total = 1 / (???)
# print("Total Parallel Resistance:", ???)
```

```python
# ==============================
# CLASSWORK 4: Conditionals
# ==============================
# Task: If series resistance > 50Ω, print "Too high!"
# Else print "Safe value"

# if ??? > 50:
#     print("Too high!")
# else:
#     print("Safe value")
```

```python
# ==============================
# CLASSWORK 5: Function Writing
# ==============================
# Task: Write a function to calculate series resistance 
# from a list of resistors.

# def total_series(resistors):
#     # Use sum() to add all resistor values
#     return ???

# print(total_series([5, 10, 15]))  # Expected: 30
```

---

### Wrap-up / Homework Challenge

Modify your function so it accepts a parameter `mode` that can be `"series"` or `"parallel"`. Depending on the mode, return the correct equivalent resistance.
