# something
just pieces
import re
f = open("E:\\python\\a.txt","r")
dd = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"  
words = ""  
for i in f:  
    for j in i:  
        if j in dd:  
            words += j 
a=re.findall('^[A-Z][A-Z]{3}([a-z])[A-Z]{3}[^A-Z]',words)   
print (a)  
