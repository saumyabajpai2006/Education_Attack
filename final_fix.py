import json
import re

# Read the notebook file
with open('analysis2/models/timeseries_model.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the escape sequence issues
content = re.sub(
    r'"c:\\Users\\LENOVO\\miniconda3\\Lib\\site-packages\\tqdm\\auto\.py:21: TqdmWarning: IProgress not found\. Please update jupyter and ipywidgets\. See https://ipywidgets\.readthedocs\.io/en/stable/user_install\.html\n",',
    '"c:\\\\\\\\Users\\\\\\\\LENOVO\\\\\\\\miniconda3\\\\\\\\Lib\\\\\\\\site-packages\\\\\\\\tqdm\\\\\\\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\\\\\\\\n",',
    content
)

content = re.sub(
    r'"  from \.autonotebook import tqdm as notebook_tqdm\n"',
    '"  from .autonotebook import tqdm as notebook_tqdm\\\\\\\\n"',
    content
)

# Remove duplicate execution_count entries by processing line by line
lines = content.split('\n')
fixed_lines = []
i = 0
in_cell = False
execution_count_seen = False

while i < len(lines):
    line = lines[i]
    
    # Check if we're entering a new cell
    if '"cell_type": "code"' in line:
        in_cell = True
        execution_count_seen = False
        fixed_lines.append(line)
    elif '"cell_type":' in line and '"cell_type": "code"' not in line:
        in_cell = False
        execution_count_seen = False
        fixed_lines.append(line)
    elif in_cell and '"execution_count":' in line and not execution_count_seen:
        execution_count_seen = True
        fixed_lines.append(line)
    elif in_cell and '"execution_count":' in line and execution_count_seen:
        # Skip duplicate execution_count
        pass
    else:
        fixed_lines.append(line)
    
    i += 1

content = '\n'.join(fixed_lines)

# Try to parse as JSON to validate
try:
    notebook = json.loads(content)
    print("JSON is valid!")
    
    # Write the cleaned content
    with open('analysis2/models/timeseries_model.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("Notebook fixed and saved successfully!")
    
except json.JSONDecodeError as e:
    print(f"JSON validation failed: {e}")
    # Write the content as is
    with open('analysis2/models/timeseries_model.ipynb', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Content written as-is")




