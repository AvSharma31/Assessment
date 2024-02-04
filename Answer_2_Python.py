from collections import Counter

def highest_frequency_word_length(input_string):
    words = input_string.split()
	
    word_frequency = Counter(words)

    highest_frequency_word = max(word_frequency, key=word_frequency.get)

    return len(highest_frequency_word)

input_string = "Hello my name is Avani Sharma"
result = highest_frequency_word_length(input_string)
print(f"Length of the highest-frequency word: {result}")