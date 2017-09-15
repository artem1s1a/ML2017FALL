import sys

text = open(sys.argv[1], 'r')
Q1 = open('Q1.txt', 'w')

words = list()
times = list()
ans = list()

for line in text:
   words = words + line.split()
for word in words:
   if word in ans:
      times[ans.index(word)] += 1
   else:
      ans.append(word)
      times.append(1)

for i in range(0, len(ans)):
   Q1.write("%s %d %d" % (ans[i], i, times[i]))
   if i != len(ans)-1:
      Q1.write("\n")

