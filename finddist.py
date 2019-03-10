import os
import shutil
import pickle
listy = os.listdir("array")

t = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82]
u = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82]
v = [0, 1, 2, 3, 5, 6, 7, 8, 10, 12, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74]

x = []
y = []
arr = []
data = [[],[],[],[],[],[]]

def comp(val):
    return len(val[1])
print(len(listy))
for i in range(0,int(len(listy))):
    line = listy[i]
    lab = int(line[9])
    # print("label is", lab)
    # if (lab == 0):
    #     r+=1
    # elif (lab ==1):
    #     g+=1
    # elif(lab == 2):
    #     b+=1
    file = open("array/"+line, 'rb')
    arr.append([lab, pickle.load(file)])
    # print (i)
    file.close()
# out = open("dist.txt", "w")
arr.sort(key = comp)
r = []
g = []
b = []
for i in range(0, len(arr)):
    print(arr[i][0], len(arr[i][1]))
    if arr[i][0] == 0:
        r.append(arr[i])
    elif arr[i][0] == 1:
        g.append(arr[i])
    elif arr[i][0] == 2:
        b.append(arr[i])

out = open("dist.txt", 'w')
print(len(r), len(g), len(b))
print(len(t), len(u), len(v))
for n in t:
    # print(n, len(r[n]))
    if os.path.isfile("newtrain/" + str(r[n][0]) + "{:04d}".format(len(r[n][1]))):
        print("newtrain/" + str(r[n][0]) + "{:04d}".format(len(r[n][1])))
        print(n)
        exit(0)
    fp = open("newtrain/" + str(r[n][0]) + "{:04d}".format(len(r[n][1])), 'wb')
    pickle.dump(r[n][1], fp)
    # data[0].append(len(r[n][1]))
    out.write(str(len(r[n][1])) + "\n")
out.write("\n\n")
for n in u:
    if os.path.isfile("newtrain/" + str(g[n][0]) + "{:04d}".format(len(g[n][1]))):
        print("newtrain/" + str(g[n][0]) + "{:04d}".format(len(g[n][1])))
        print(n)
        exit(0)
    fp = open("newtrain/" + str(g[n][0]) + "{:04d}".format(len(g[n][1])), 'wb')
    pickle.dump(g[n][1], fp)
    # data[1].append(len(g[n][1]))
    out.write(str(len(g[n][1])) + "\n")
out.write("\n\n")
for n in v:
    if os.path.isfile("newtrain/" + str(b[n][0]) + "{:04d}".format(len(b[n][1]))):
        print("newtrain/" + str(b[n][0]) + "{:04d}".format(len(b[n][1])))
        print(n)
        exit(0)
    fp = open("newtrain/" + str(b[n][0]) + "{:04d}".format(len(b[n][1])), 'wb')
    pickle.dump(b[n][1], fp)
    # data[2].append(len(b[n][1]))
    out.write(str(len(b[n][1])) + "\n")
out.write("\n\n")
print("Finished training set data")
for i in range(0, 91):
    if i not in t:
        tes = r[i]
        fp = open("newtest/" + str(tes[0]) + "{:04d}".format(len(tes[1])), 'wb')
        pickle.dump(tes[1], fp)
        # data[3].append(len(tes[1]))
        out.write(str(len(tes[1])) + "\n")
out.write("\n\n")
for i in range(0, 86):
    if i not in u:
        tes = g[i]
    fp = open("newtest/" + str(tes[0]) + "{:04d}".format(len(tes[1])), 'wb')
    pickle.dump(tes[1], fp)
    # data[4].append(len(tes[1]))
    out.write(str(len(tes[1])) + "\n")
out.write("\n\n")
for i in range(0, 75):
    if i not in v:
        tes = b[i]
    fp = open("newtest/" + str(tes[0]) + "{:04d}".format(len(tes[1])), 'wb')
    pickle.dump(tes[1], fp)
    # data[5].append(len(tes[1]))
    out.write(str(len(tes[1])) + "\n")
out.write("\n\n")
# print(r, g, b)

# out = open("dist.txt", 'w')
# for i in range (0, len(data)):
    # out.write(str(data[0]) + ", " + data[2]) + ", " + data[3]) + ", " + data[4]) + ", " + data[0]) + ", " + data[0]) + ", " + )