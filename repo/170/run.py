import sys, os

start, finish = int(sys.argv[1]), int(sys.argv[2])
for i in range(start, finish+1):
	os.system("python main.py " + str(i) + " " + str(i))