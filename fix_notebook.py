import json
import re

# Read the notebook file
with open('analysis2/models/timeseries_model.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the JSON structure by removing empty lines and fixing the structure
# Remove empty lines that were left after removing merge conflict markers
content = re.sub(r'\n\s*\n\s*\n', '\n', content)

# Fix the specific issue with the warning message structure
content = re.sub(
    r'"text": \[\s*\n\s*"c:\\\\Users\\\\LENOVO\\\\miniconda3\\\\Lib\\\\site-packages\\\\tqdm\\\\auto\.py:21: TqdmWarning: IProgress not found\. Please update jupyter and ipywidgets\. See https://ipywidgets\.readthedocs\.io/en/stable/user_install\.html\\n",\s*\n\s*"C:\\\\Users\\\\saumy\\\\AppData\\\\Roaming\\\\Python\\\\Python312\\\\site-packages\\\\tqdm\\\\auto\.py:21: TqdmWarning: IProgress not found\. Please update jupyter and ipywidgets\. See https://ipywidgets\.readthedocs\.io/en/stable/user_install\.html\\n",\s*\n\s*"  from \.autonotebook import tqdm as notebook_tqdm\\n"\s*\n\s*"source": \[',
    '"text": [\n      "c:\\\\Users\\\\LENOVO\\\\miniconda3\\\\Lib\\\\site-packages\\\\tqdm\\\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\\n",\n      "  from .autonotebook import tqdm as notebook_tqdm\\n"\n     ]\n    },\n    {\n     "name": "stdout",\n     "output_type": "stream",\n     "text": []\n    }\n   ],\n   "source": [',
    content
)

# Write the cleaned content back
with open('analysis2/models/timeseries_model.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print("Notebook cleaned successfully!")




