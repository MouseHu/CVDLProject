import sys
import numpy as np
data=open("shit.txt")
shit=set()
label=0
shitshit={}
for i in data.readlines():
	rgb=i.rstrip("\n").split("	")
	r=int(float(rgb[0])*255+0.5)
	g=int(float(rgb[1])*255+0.5)
	b=int(float(rgb[2])*255+0.5)
	if (r,g,b) not in shit:
		shit.add((r,g,b))
		shitshit[(r,g,b)]=label
		label+=1
print(shitshit)
print(len(shit))
