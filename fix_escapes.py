import re

# Read the notebook file
with open('analysis2/models/timeseries_model.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the specific escape sequence issue
content = re.sub(
    r'"c:\\Users\\LENOVO\\miniconda3\\Lib\\site-packages\\tqdm\\auto\.py:21: TqdmWarning: IProgress not found\. Please update jupyter and ipywidgets\. See https://ipywidgets\.readthedocs\.io/en/stable/user_install\.html\\n",',
    '"c:\\\\\\\\Users\\\\\\\\LENOVO\\\\\\\\miniconda3\\\\\\\\Lib\\\\\\\\site-packages\\\\\\\\tqdm\\\\\\\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\\\\\\\\n",',
    content
)

# Fix the autonotebook import line
content = re.sub(
    r'"  from \.autonotebook import tqdm as notebook_tqdm\\n"',
    '"  from .autonotebook import tqdm as notebook_tqdm\\\\\\\\n"',
    content
)

# Write the fixed content
with open('analysis2/models/timeseries_model.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print("Escape sequences fixed!")


