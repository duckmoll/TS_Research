import pprint
import pandas as pd
import re

def extract_content(input_string, regex_pattern):
    """Extracts content from a string using a regex pattern."""
    match = re.search(regex_pattern, input_string)
    # Check if a match was found and extract the captured group (group 1)
    if match:
        return match.group(1)

def parse_forecast_data(filepath):
    """
    Parses a text file where data is grouped in two-line entries.

    Args:
        filepath (str): The path to the text file.

    Returns:
        list: A list of dictionaries, where each dictionary represents
              a two-line data group from the file. Returns an empty
              list if the file is not found or is empty.
    """
    parsed_data = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return []

    # Filter out empty or whitespace-only lines
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    # Process the lines in pairs (groups of two)
    # The first line is the identifier, the second contains metrics.
    for i in range(0, len(non_empty_lines), 2):
        # Ensure we have a pair of lines to process
        if i + 1 < len(non_empty_lines):
            identifier = non_empty_lines[i]
            identifier = identifier.replace("national_illness", "illness")
            metrics_line = non_empty_lines[i+1]

            dataset = identifier.replace("long_term_forecast_", "").split('_')[0]
            pred_len = identifier.replace("long_term_forecast_", "").split('_')[2]
            # mode = identifier.replace("long_term_forecast_", "").split('_')[3]
            pattern = r'.*_(?:\d+)_(.*?)_PatchTST'
            mode = extract_content(identifier, pattern).replace('multipleallmoemarktopk_', "")
            mode = "NoEncoder" if "Noencoder" in identifier else "Encoder"

            if dataset in ["exchange", "illness"]:
                continue

            data_entry = {
                "dataset": dataset,
                "pred_len": pred_len,
                "mode": mode,
            }

            # Parse the metrics line (e.g., "mse:0.0606, mae:0.1899, dtw:Not calculated")
            metrics = metrics_line.split(',')
            for metric in metrics:
                try:
                    key, value = metric.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'dtw':
                        continue

                    # Try to convert numeric values to float
                    try:
                        data_entry[key] = float(value)
                    except ValueError:
                        data_entry[key] = value
                except ValueError:
                    # Handle cases where a metric might not be in key:value format
                    print(f"Warning: Could not parse metric '{metric}' in line {i*2+2}")


            parsed_data.append(data_entry)

    return parsed_data

if __name__ == "__main__":
    # The name of the file to parse
    filename = 'result_encoder.txt'

    # Parse the data from the file
    forecast_results = parse_forecast_data(filename)

    # Pretty-print the parsed data
    if forecast_results:
        print(f"Successfully parsed {len(forecast_results)} entries from '{filename}'.")
        print("Here is the parsed data:")
        pprint.pprint(forecast_results)

    result_df = pd.DataFrame(forecast_results)
    column_order = ["raw", "single", "multiple", "moe", "multiple_all", "multiple_all_moe", "multiple_all_moe_mark"]
    column_order = ["1", "3", "5", "7", "9"]
    column_order = ["Encoder", "NoEncoder"]
    row_order = ["96", "192", "336", "720"]
    result_df = result_df.set_index(["dataset", "pred_len", "mode"])
    print(result_df)
    result_df.to_excel('result_encoder_stats.xlsx')
    result_df = result_df.unstack("mode").reindex(columns=column_order,
                        level=-1).reindex(index=row_order, level=-1)
    print(result_df)
    result_df.to_excel('result_encoder_stats.xlsx')