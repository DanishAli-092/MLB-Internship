# Day 2 - Dictionaries Practice
#MLB Internship

# 1. Create a student record dictionary
student = {
    "name": "Danish Ali",
    "roll_no": "F22-123",
    "age": 21,
    "course": "AI"
}
print("Student Record:", student)

# 2. Calculate average marks of students
students_marks = {
    "Ali": [80, 90, 70],
    "Sara": [60, 75, 85],
    "Bilal": [95, 85, 90]
}

def average_marks(marks_dict):
    averages = {}
    for name, marks in marks_dict.items():
        total = sum(marks)
        avg = total / len(marks)
        averages[name] = avg
    return averages

print("\nMarks dictionary:", students_marks)
print("Averages:", average_marks(students_marks))

# 3. Count frequency of words in a sentence
def word_frequency(sentence):
    words = sentence.lower().split()
    freq = {}
    for word in words:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq

sentence = "the quick brown fox jumps over the lazy dog the fox was quick"
print("\nSentence:", sentence)
print("Word Frequency:", word_frequency(sentence))