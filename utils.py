def clean_broken_words(text, paragraphs):
    # Remove extra dots or ellipses and identify partial words
    ellipsis_pattern = re.compile(r'\b\w*\…\w*\b')
    broken_words = ellipsis_pattern.findall(text)

    if broken_words:
        # Clean the text by removing the ellipses
        cleaned_text = re.sub(r'\…+', '', text)
        question_words = cleaned_text.split()

        # Split words from paragraphs
        for para in paragraphs:
            paragraph_words = para['text'].split()
            for i, word in enumerate(question_words):
                # If word was broken and is not in the paragraph, look for a match
                if word not in paragraph_words:
                    for para_word in paragraph_words:
                        # If part of the word is found in the paragraph word, replace it
                        if word in para_word:
                            question_words[i] = para_word

        # Join the cleaned words back into the sentence
        cleaned_text = ' '.join(question_words)
        return cleaned_text
    return text

def table_row_to_text( header, row):
    '''
    Convert a table row to text using the provided header.
    Constructs descriptive sentences from header and row values.
    '''
    res = ""

    # Add the first header item if it exists
    # for index in title:
    #   if index and isinstance(index, str):
    #     res += (index + " ")

    # Iterate through the header and row, skipping the first item
    for head, cell in zip(header[1:], row[1:]):
        if cell:  # Check if the cell is not empty
            res += f"the {row[0]} of {head} is {cell} ; "

    res = remove_space(res)
    return res.strip()


def remove_space(text_in):
    '''Remove extra spaces from the input text.'''
    return " ".join(text_in.split())  # Simplified to remove extra spaces


# Function to find answer start positions when not provided
def find_answer_positions(context, answer):
    # Debugging: Print the context and answer for visibility
    print(f"Context: {context[:100]}...")  # Print the first 100 characters for brevity
    print(f"Answer: {answer}")

    # Find the position of the answer in the context


    # Check if the answer contains '|', indicating multiple parts
    if '|' in answer:
        answer_parts = [part.strip() for part in answer.split('|')]  # Split and strip each part
        # Find the position of each part in the context
        positions = []
        for part in answer_parts:
            start_idx = context.find(part)
            if start_idx == -1:
                continue
            end_idx = start_idx + len(part)
            positions.append((start_idx, end_idx))
    else:
        start_idx = context.find(answer)  # Treat as a single answer

    if start_idx == -1:
        # raise ValueError(f"Answer '{answer}' not found in context.")
        start_idx = 0
    end_idx = start_idx + len(answer)
    return start_idx, end_idx

# Function to detect if a question involves a mathematical operation
def detect_math_operation(question):
    # Keywords for different operations
    operations = {
        'sum': ['sum', 'add', 'total'],
        'difference': ['difference', 'subtract', 'minus'],
        'compare': ['compare', 'greater', 'lesser'],
        'max': ['max', 'maximum', 'highest'],
        'min': ['min', 'minimum', 'lowest'],
        'percentage': ['percentage', 'percent', 'growth']
    }

    for operation, keywords in operations.items():
        for keyword in keywords:
            if keyword in question.lower():
                return operation
    return None