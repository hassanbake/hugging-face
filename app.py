from transformer import Classification, Generator

def main():
    classification = Classification()

    result = classification.classify(["I have been waiting for Huugging face course my whole life", 
                                   "I hate this so much."])
    print(result)

    result = classification.classify("This is a course about Transformer libarary.",
                                  ["education","politics","business"])
    print(result)

    result = Generator.generator("In this course, we will teach you how to")
    print(result)

if __name__ == "__main__":
    main()