import string as s
def count_characters(string):
    for char in string:
        if not char.isalpha():
            return False
        return True
Test_case = int(input())
for i in range(0, Test_case):
    string = input("")
    character_list=set(s.punctuation+s.whitespace+s.digits)
    character_list.remove(" ")
    replacement_character=""
    new_string=""
    character_count = {}
    if count_characters(string)==False:
        for character in string:
            if character in character_list:
                new_string+= replacement_character
            else:
                new_string+=character
    split_words = new_string.split()
    for word in split_words:
        if word in character_count:
            character_count[word * 2] = len(word)
        else:
            character_count[word] = len(word)
    values_list = character_count.values()
    values_string = ','.join(map(str, values_list))
    print(values_string)


