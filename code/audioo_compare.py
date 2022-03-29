import wave
def compare():
    w_one = wave.open('file_one', 'r')
    w_two = wave.open('file_two', 'r')

    if w_one.readframes() == w_two.readframes():
        print('exactly the same')
    else:
        print('not a match')