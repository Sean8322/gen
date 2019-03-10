path = input("PLease enter filepath ")

file = open(path, 'r')
line = file.readline()
while line:
    if line.find("poch") != -1:
        print(line)
    if line.find("loss") != -1:
        print(line)
    line = file.readline()