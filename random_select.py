import os
import argparse
import random

TRAIN_PERCENTAGE = 0.7

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)


args = parser.parse_args()


file_list = []
for file in os.listdir(args.input):
    if file.endswith(".txt") and file != "classes.txt":
        file_list.append(os.path.abspath(os.path.join(args.input, file)).replace("/labels", "").replace(".txt", ".jpg"))


total = len(file_list)

eval_set = []

while (len(file_list)/float(total) > TRAIN_PERCENTAGE):
    selection = random.choice(file_list)
    eval_set.append(selection)
    file_list.remove(selection)

f1 = open(args.output+"/training.txt", "w+")
f2 = open(args.output+"/eval.txt", "w+")

for item in eval_set:
    f2.write(item + "\n")

for item in file_list:
    f1.write(item+"\n")


print ("evaluation:",len(eval_set))
print ("training:",len(file_list))
