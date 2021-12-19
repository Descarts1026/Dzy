

total = 0
ff = 0
with open("time.txt","r") as f:
    alllines = f.readlines()
    for line in alllines:
        total +=1
        current = float(line.split(" ")[-1].strip())
        ff+=current

print("fps:",1/(ff/total))
#fps: 6.2620286030863435