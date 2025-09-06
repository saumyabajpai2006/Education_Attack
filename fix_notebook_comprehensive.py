import json
import re

# Read the notebook file
with open('analysis2/models/timeseries_model.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix duplicate execution_count entries by removing the duplicates
# This regex finds patterns where execution_count appears multiple times in the same cell
content = re.sub(
    r'("execution_count": \d+,)\s*\n\s*("id": "[^"]+",)\s*\n\s*("metadata": {},\s*)\n\s*("outputs": \[\s*)\n\s*({[^}]*"execution_count": \d+,[^}]*})',
    r'\1\n   \2\n   \3\n   \4\n     \5',
    content,
    flags=re.MULTILINE | re.DOTALL
)

# Fix the specific warning message structure issue
content = re.sub(
    r'"text": \[\s*\n\s*"c:\\\\Users\\\\LENOVO\\\\miniconda3\\\\Lib\\\\site-packages\\\\tqdm\\\\auto\.py:21: TqdmWarning: IProgress not found\. Please update jupyter and ipywidgets\. See https://ipywidgets\.readthedocs\.io/en/stable/user_install\.html\\n",\s*\n\s*"  from \.autonotebook import tqdm as notebook_tqdm\\n"\s*\n\s*\]\s*\n\s*},\s*\n\s*{\s*\n\s*"name": "stdout",\s*\n\s*"output_type": "stream",\s*\n\s*"text": \[\]\s*\n\s*}\s*\n\s*\]\s*,\s*\n\s*"source": \[',
    '"text": [\n      "c:\\\\Users\\\\LENOVO\\\\miniconda3\\\\Lib\\\\site-packages\\\\tqdm\\\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\\n",\n      "  from .autonotebook import tqdm as notebook_tqdm\\n"\n     ]\n    },\n    {\n     "name": "stdout",\n     "output_type": "stream",\n     "text": []\n    }\n   ],\n   "source": [',
    content
)

# Remove any remaining empty lines that might cause issues
content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

# Try to parse and reformat the JSON to ensure it's valid
try:
    notebook = json.loads(content)
    # Write back with proper formatting
    with open('analysis2/models/timeseries_model.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("Notebook cleaned and reformatted successfully!")
except json.JSONDecodeError as e:
    print(f"JSON parsing error: {e}")
    # Write the content as is if JSON parsing fails
    with open('analysis2/models/timeseries_model.ipynb', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Content written as-is due to JSON parsing error")

