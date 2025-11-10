# Python Foundations for Electrical Engineers

**Scope**: Absolute basics → control flow → functions → up to **core data structures** (string, list, tuple, set, dict) + light CSV I/O.

---

## 0) Why Python for EE?

Python হল আপনার "lab assistant"—স্ক্রিপ্ট লিখে বারবারের কাজ automate করে, data ingest করে, basic math/logic চালিয়ে দ্রুত **simulate → analyze → visualize** করা যায়।

**Use-cases (EE-flavored)**: sensor logging, signal pre-processing, CSV (Comma Separated Values) / DAQ (Dats Acquisition) parsing, batch circuit calculations, small optimization, plotting.

---

## 1) Getting Ready to Run Code

* **Ways to run**: PyCharm, VS Code, REPL (Python shell), `.py` script, Jupyter/Colab notebook. 
* **Files & folders**: keep a project folder: `ee101/` → `data/`, `notebooks/`, `src/`.
* **Version**: Prefer Python 3.11+ if possible.

> **Tip (Bangla)**: প্রথমে notebook (e.g., Google Colab) এ practice করলে দ্রুত feedback পাওয়া যায়। পরে `.py` script এ একই logic রাখুন—production-style runs-এর জন্য ভালো।

---

## 2) Numbers, Variables, and Units

Python supports `int`, `float`, `bool`, `str`.

```python
# ints and floats
v = 5          # volts
r = 220        # ohms
i = v / r      # amps (float)
print("Current (A):", i)

# scientific notation
c = 4.7e-6  # 4.7 microfarads
print("Capacitance (F):", c)

# underscores for readability
big = 1_000_000
print("One million:", big)
```

> **Bangla**: variable নাম snake_case এ দিন: `sample_rate_hz`, `cutoff_freq`, ইত্যাদি। Units মন্তব্যে লিখুন—code পড়া সহজ হবে।

---

## 3) Strings & Basic Formatting (quick)

```python
name = "EE Lab"
reading = 3.25789
print(f"{name} avg voltage = {reading:.3f} V")  # f-string with 3 decimals
```

> **Bangla**: `f"...{expr}..."` ফরম্যাটিং করলে measurement neatly দেখানো যায়।

---

## 4) Control Flow — If/Elif/Else

```python
v = 9
r = 470
current = v / r

if current > 0.02:
    print("Warning: current high!")
elif current < 0.001:
    print("Current very low")
else:
    print("Current in acceptable range")
```

> **Bangla**: `if/elif/else` মানে ধারাবাহিক শর্ত পরীক্ষা। Truthiness: `0`, `""`, `[]`, `{}` → **False**; অন্য সবকিছু সাধারণত **True**।

---

## 5) Loops — The Beating Heart (Comprehensive)

### 5.1 `for` loop over iterables

```python
resistors = [100, 220, 330]
series = 0
for r in resistors:
    series += r
print("Series (ohms):", series)
```

**Patterns**:

* **`range(n)`**: 0…n-1 বার iterate
* **`enumerate(iterable)`**: index + value
* **`zip(a, b)`**: pair-wise iterate

```python
# range
for k in range(5):
    print("k:", k)

# enumerate
vals = [3.1, 3.3, 3.2]
for idx, v in enumerate(vals):
    print("index:", idx, "value:", v)

# zip
times = [0, 1, 2]
for t, v in zip(times, vals):
    print("t:", t, "V:", v)
```

> **Bangla**: `enumerate` (এনিউমারেট) index দরকার হলে elegant। `zip` (জিপ) একাধিক list একসাথে ধরতে সাহায্য করে।

### 5.2 `while` loop

```python
count = 0
while count < 3:
    print("count:", count)
    count += 1
```

> **Bangla সতর্কতা**: `while` এর stop condition ভুল হলে **infinite loop** হতে পারে। Counter update ভুলবেন না।

### 5.3 `break`, `continue`, loop `else`

```python
# find first value above threshold
vals = [0.8, 0.9, 1.2, 0.95]
threshold = 1.0
for v in vals:
    if v > threshold:
        print("Found:", v)
        break
else:
    print("No value above threshold")
```

> **Bangla**: `break` লুপ থামায়; `continue` current iteration skip করে; `for ... else` শুধুমাত্র তখনই `else` চালায় যখন `break` হয়নি।

### 5.4 Comprehensions (list/set/dict)

```python
# List comprehension: square of even indices
sq = [k*k for k in range(10) if k % 2 == 0]
print("squares:", sq)

# Set comprehension: unique rounded voltages
raw_v = [3.141, 3.142, 3.139]
uniq = {round(x, 2) for x in raw_v}
print("unique:", uniq)

# Dict comprehension: index→value map
mp = {i: v for i, v in enumerate(raw_v)}
print("map:", mp)
```

> **Bangla**: comprehension সংক্ষিপ্ত, কিন্তু overuse করলে পড়া কঠিন—simple রাখুন।

---

## 6) Functions — Reusable Brains

```python
def series_eq(resistors):
    total = 0
    for r in resistors:
        total += r
    return total

def parallel_eq(resistors):
    return 1 / sum(1/r for r in resistors)

print("Series:", series_eq([100, 200, 300]))
print("Parallel:", parallel_eq([100, 200, 300]))
```

**Best practices**

* Single, clear purpose; small.
* Use **type hints** for clarity.
* Document behavior with a **docstring**.

```python
def mv_avg(x: list[float], window: int) -> list[float]:
    """Return moving-average of list x with given window size."""
    out = []
    for i in range(len(x) - window + 1):
        out.append(sum(x[i:i+window]) / window)
    return out
```

> **Bangla**: function (ফাংশন) reuse বাড়ায়, bugs কমায়। Type hint (টাইপ হিন্ট) future you/teammate-এর জন্য gift।

---

## 7) Core Data Structures (Deep but Friendly)

### 7.1 String

* Immutable (change-in-place নয়); slicing supports `[start:stop:step]`.

```python
s = "EE-101"
print(s[0:2])       # 'EE'
print(s.split("-")) # ['EE', '101']
```

> **Bangla**: string (স্ট্রিং) data cleaning-এ heavy use—`strip`, `lower`, `replace` কাজে লাগে।

### 7.2 List (ordered, mutable)

```python
voltages = [3.1, 3.3, 3.2]
voltages.append(3.4)
voltages[1] = 3.25
x = voltages[:]          # shallow copy
print("V list:", voltages)
print("Copy:", x)
```

Common ops: `append`, `extend`, `insert`, `pop`, `remove`, `sort`, `reverse`, slicing.

> **Bangla সতর্কতা**: list-of-lists copy করলে deep vs shallow বুঝুন; `import copy; copy.deepcopy(...)` প্রয়োজন হতে পারে।

### 7.3 Tuple (ordered, immutable)

```python
reading = (0.0, 3.30)   # (time_s, voltage_v)
print(reading[1])
```

Use for fixed-size records; safe as dict keys.

### 7.4 Set (unique, unordered)

```python
sensors = ["A1", "A2", "A2", "A3"]
unique = set(sensors)
print("Unique:", unique)
```

Fast membership tests, math ops: `|` union, `&` intersection, `-` difference.

### 7.5 Dict (hash map: key→value)

```python
config = {"sample_rate_hz": 100, "sensor": "A1"}
print(config["sample_rate_hz"])

# safe get with default
sr = config.get("sample_rate_hz", 50)
print("SR:", sr)

# nested
row = {"t": 0, "V": 3.1, "meta": {"unit": "V", "loc": "Bench-1"}}
print(row["meta"]["unit"])
```

Iteration patterns:

```python
for k, v in config.items():
    print(k, v)
```

> **Bangla**: dict (ডিকশনারি) হল labeled data container—CSV row, config, lookup table—সবকিছুর প্রাণ।

### 7.6 Choosing the Right Structure

* **Sequence you mutate** → List
* **Fixed-size record** → Tuple
* **Uniqueness / membership** → Set
* **Labeled fields / fast lookup** → Dict

---

## 8) Light File & CSV I/O

```python
import csv
# write
with open("vt.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "voltage_v"])
    for t, v in [(0,3.1),(1,3.3),(2,3.2)]:
        w.writerow([t, v])

# read
T, V = [], []
with open("vt.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        T.append(float(row["time_s"]))
        V.append(float(row["voltage_v"]))

print("n:", len(V), "min:", min(V), "max:", max(V))
```

> **Bangla**: DAQ থেকে CSV আকারে data এলে `csv` module দিয়ে reliably parse করুন; header names পরিষ্কার রাখুন।

---

## 9) EE Mini-Examples

### 9.1 RC step response (Euler approx)

```python
# RC: dv/dt = (Vin - v)/RC  → v_{k+1} ≈ v_k + dt*(Vin - v_k)/(R*C)
Vin = 5.0
R = 1_000
C = 4.7e-6
v = 0.0
T, V = [], []

dt = 0.001
for n in range(2000):
    t = n*dt
    v = v + dt * (Vin - v) / (R*C)
    T.append(t); V.append(v)
print("Final approx V:", V[-1])
```

> **Bangla**: physics-to-code mapping (মডেল→ডিসক্রেট স্টেপ) খুব গুরুত্বপূর্ণ—loop দিয়ে টাইম-ডোমেইন simulation।

### 9.2 Resistor color code lookup

```python
colors = {"black":0,"brown":1,"red":2,"orange":3,"yellow":4,
          "green":5,"blue":6,"violet":7,"grey":8,"white":9}

def value_from_bands(a, b, mult):
    base = 10*colors[a] + colors[b]
    return base * (10 ** colors[mult])

print("10k example:", value_from_bands("brown","black","orange"))
```

### 9.3 Moving average filter

```python
raw = [3.1, 3.3, 3.2, 3.4, 3.3]
print("MA-3:", mv_avg(raw, 3))
```

---

## 10) Common Pitfalls & Debugging Habits

* Float rounding: compare with tolerances, e.g., `abs(a-b) < 1e-9`.
* Off-by-one in loops: check `range()` endpoints.
* Mutable defaults in functions: never use `def f(x, bag=[])`; use `None` then create.
* Print small checkpoints; keep variables short but meaningful.

> **Bangla**: bug ধরা সহজ করতে inputs/outputs ছোট রাখুন, এবং উদাহরণ hard-code করে শুরু করুন—পরে generalize করুন।

---

## 11) Practice Set (with brief solutions)

### Q1: Series vs Parallel

**Task**: `mode` = `"series"` বা `"parallel"` অনুযায়ী equivalent resistance return করুন.

```python
def req(values, mode="series"):
    if mode == "series":
        return sum(values)
    elif mode == "parallel":
        return 1/sum(1/x for x in values)
    else:
        raise ValueError("unknown mode")
```

### Q2: Extract max voltage timestamp

**Task**: `rows = [{"t":..., "V":...}, ...]` থেকে max-`V` এর `t` দিন।

```python
def t_at_vmax(rows):
    best = max(rows, key=lambda r: r["V"])
    return best["t"]
```

### Q3: Unique sensor IDs (sorted)

```python
def unique_sorted(ids):
    return sorted(set(ids))
```

### Q4: CSV mean/min/max one-liner-ish (readability first)

```python
import csv

def summary_csv(path):
    T, V = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            T.append(float(row["time_s"]))
            V.append(float(row["voltage_v"]))
    return {"n": len(V), "mean": sum(V)/len(V), "min": min(V), "max": max(V)}
```

### Q5: Map sensor→list of readings

```python
def group_by_sensor(pairs):
    # pairs like [("A1", 3.1), ("A2", 3.0), ("A1", 3.2)]
    table: dict[str, list[float]] = {}
    for sid, v in pairs:
        table.setdefault(sid, []).append(v)
    return table
```

---

## 12) What’s Next (after Data Structures)

* **NumPy** for fast arrays/vectorization.
* **Pandas** for tables/time-series.
* **Matplotlib** for plots.
* **Scipy** for signal processing.

> **Bangla**: আজ যা শিখলেন—এগুলোই foundation। এরপর NumPy/Pandas নিলে গবেষণার workflow কয়েকগুণ দ্রুত হবে।

---

### One-page Cheat Sheet (printable)

* **If/Elif/Else**: branch by condition; "truthy"/"falsy" ভেবে দেখুন।
* **For loop**: iterate over list/tuple/set/dict; `enumerate`, `zip` handy.
* **While loop**: condition-driven; infinite loop এ সাবধান।
* **Comprehension**: concise create/transform; overuse করবেন না।
* **List** (mutable seq), **Tuple** (immutable record), **Set** (unique bag), **Dict** (key→value).
* **CSV**: `csv.DictReader` + clean headers; numbers `float()` করুন।
* **Functions**: single purpose, type hints, docstring.

*Stay curious. Build tiny tools. Let code carry the load.*
