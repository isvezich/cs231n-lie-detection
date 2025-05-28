import csv
import os

def filter_csv_by_prefix(csv_input_path, target_directory, csv_output_path):
    matched_rows = []

    # Read original CSV
    with open(csv_input_path, newline='') as infile:
        reader = csv.reader(infile)
        rows = [row for row in reader if row]  # Skip empty rows

    # Get list of filenames in target directory
    filenames = os.listdir(target_directory)

    # Check for matching prefixes
    for row in rows:
        prefix = row[0]
        if any(filename.startswith(prefix) for filename in filenames):
            matched_rows.append(row)

    # Write matched rows to a new CSV
    with open(csv_output_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(matched_rows)

    print(f"Written {len(matched_rows)} matching rows to {csv_output_path}")

if __name__ == "__main__":
    input_csv = "test_fold3.csv"               # Path to your input CSV
    directory = "./data/videos"      # Path to your directory with files
    output_csv = "matched_rows.csv"       # Output CSV path

    filter_csv_by_prefix(input_csv, directory, output_csv)