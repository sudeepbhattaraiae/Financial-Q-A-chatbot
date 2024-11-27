import re
import pandas as pd
from io import StringIO


def truncate_column_name(name, max_length=50):
    if isinstance(name, float) or pd.isna(name):
        name = "unnamed_column"
    else:
        # Remove special characters and replace spaces with underscores
        name = re.sub(r'[^\w\s]', '', name)
        if name == '':
            name = "unnamed_column"
        name = name.strip().replace(' ', '_')
        # Ensure the name starts with a letter or underscore
        if not name[0].isalpha() and name[0] != '_':
            name = 'col_' + name
        # Truncate if necessary
        if len(name) > max_length:
            name = name[:max_length].rstrip('_')
    return name

def get_unique_column_names(df):
    seen = set()
    unique_names = []
    for col in df.columns:
        truncated = truncate_column_name(col)
        if truncated in seen:
            i = 1
            while f"{truncated}_{i}" in seen:
                i += 1
            truncated = f"{truncated}_{i}"
        seen.add(truncated)
        unique_names.append(truncated)
    return unique_names
 

def extract_tables_from_markdown(markdown_text):
    table_pattern = r'(\|.*\|(?:\n\|.*\|)+)'
    table_matches = re.findall(table_pattern, markdown_text, re.MULTILINE)

    tables = []
    for table_match in table_matches:
        lines = table_match.strip().split('\n')
        num_columns = len(lines[0].split('|')) - 2
        
        processed_lines = []
        for line in lines:
            cells = line.split('|')[1:-1]
            processed_line = '|'.join([''] + cells[:num_columns] + [''])
            processed_lines.append(processed_line)
        
        processed_table = '\n'.join(processed_lines)
        
        try:
            df = pd.read_table(StringIO(processed_table), sep='|', header=0, skipinitialspace=True)
            df = df.dropna(axis=1, how='all')
            df = df.apply(lambda x: x.strip() if isinstance(x, str) else x)
            
            if df.iloc[0].str.contains('-').all():
                df = df.iloc[1:]
            
            df.columns = [col.strip() for col in df.columns]
            df = df.reset_index(drop=True)

            df.columns = get_unique_column_names(df)

            for col in df.columns:
                df[col] = df[col].replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            tables.append(df)
        except Exception as e:
            print(f"Error processing table: {e}")
            print(f"Problematic table content:\n{processed_table}")

    return tables
    

def extract_text_from_markdown(markdown_text):
    # Remove code blocks
    markdown_text = re.sub(r'```[\s\S]*?```', '', markdown_text)
    # Remove inline code
    markdown_text = re.sub(r'`[^`\n]+`', '', markdown_text)
    
    # Remove links
    markdown_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_text)
    
    # Remove headers
    markdown_text = re.sub(r'#+\s*', '', markdown_text)
    
    # Remove bold and italic formatting
    markdown_text = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', markdown_text)
    
    # Remove bullet points and numbered lists
    markdown_text = re.sub(r'^\s*[-*+]\s+', '', markdown_text, flags=re.MULTILINE)
    markdown_text = re.sub(r'^\s*\d+\.\s+', '', markdown_text, flags=re.MULTILINE)
    
    # Remove horizontal rules
    markdown_text = re.sub(r'^-{3,}$', '', markdown_text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    markdown_text = markdown_text.strip()
    
    return markdown_text