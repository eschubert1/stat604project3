import csv
file_path_in = "data/raw_data/exp1_results.txt"
file_path_out = "data/processed_data/exp1_clean.txt"

rows = []
with open(file_path_in, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header lineheader = next(reader)  # skip header line
    i_before_weight = header.index("Initial Weights")
    i_after_weight = header.index("Day 5 Weight")
    for row in reader:
        # assume last two entries are numeric
        *rest, x, y = row
        before_weight = float(row[i_before_weight])
        after_weight = float(row[i_after_weight])
        weight_delta = after_weight - before_weight
        avg = (float(x) + float(y)) / 2
        rows.append(rest + [avg] + [weight_delta])
    # write back with new header
new_header = header[:-2] + ["Average Score"] + ["Weight Delta"]

with open(file_path_out, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(new_header)
    writer.writerows(rows)


file_path_in = "data/raw_data/exp2_smell_test_results.txt"
file_path_out = "data/processed_data/exp2_clean.txt"

rows = []
with open(file_path_in, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header lineheader = next(reader)  # skip header line
    i_before_weight = header.index("Initial Weights")
    i_after_weight = header.index("Day 5 Weight")
    for row in reader:
        # assume last two entries are numeric
        *rest, x, y, z = row
        before_weight = float(row[i_before_weight])
        after_weight = float(row[i_after_weight])
        weight_delta = after_weight - before_weight
        avg = (float(x) + float(y) + float(z)) / 3
        rows.append(rest + [avg] + [weight_delta])
    # write back with new header
new_header = header[:-3] + ["Average Score"] + ["Weight Delta"]

with open(file_path_out, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(new_header)
    writer.writerows(rows)
