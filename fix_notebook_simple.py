import re

# Read the notebook file
with open('analysis2/models/timeseries_model.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the escape sequence issue in the warning message
content = re.sub(
    r'"c:\\\\Users\\\\LENOVO\\\\miniconda3\\\\Lib\\\\site-packages\\\\tqdm\\\\auto\.py:21: TqdmWarning: IProgress not found\. Please update jupyter and ipywidgets\. See https://ipywidgets\.readthedocs\.io/en/stable/user_install\.html\\n",',
    '"c:\\\\\\\\Users\\\\\\\\LENOVO\\\\\\\\miniconda3\\\\\\\\Lib\\\\\\\\site-packages\\\\\\\\tqdm\\\\\\\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\\\\n",',
    content
)

# Fix the autonotebook import line
content = re.sub(
    r'"  from \.autonotebook import tqdm as notebook_tqdm\\n"',
    '"  from .autonotebook import tqdm as notebook_tqdm\\\\n"',
    content
)

# Remove duplicate execution_count entries by keeping only the first occurrence in each cell
# This is a more targeted approach
lines = content.split('\n')
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if '"execution_count":' in line and i + 1 < len(lines) and '"execution_count":' in lines[i + 1]:
        # Skip the duplicate
        i += 1
        continue
    fixed_lines.append(line)
    i += 1

content = '\n'.join(fixed_lines)

# Write the fixed content
with open('analysis2/models/timeseries_model.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print("Notebook fixed successfully!")







