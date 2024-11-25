import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text into words
words = word_tokenize(text)

# Perform POS tagging
pos_tags = pos_tag(words)

# Print the POS tags
print("Word\tPOS Tag")
for word, tag in pos_tags:
    print(f"{word}\t{tag}")
