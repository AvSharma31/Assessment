from collections import Counter

def highest_frequency_word_length(input_string):
    words = input_string.split()
	
    word_frequency = Counter(words)

    highest_frequency_word = max(word_frequency, key=word_frequency.get)

    return len(highest_frequency_word)

input_string_1 = "write write write all the number from from from 1 to 100"
result_1 = highest_frequency_word_length(input_string_1)
print(f"Length of the highest-frequency word: {result_1}")

input_string_2 = "Hello world! Hello Python world! world! world!"
result_2 = highest_frequency_word_length(input_string_2)
print(f"Length of the highest-frequency word: {result_2}")
