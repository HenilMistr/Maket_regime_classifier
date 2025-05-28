import os 
print([f for f in os.listdir() if f.endswith(".csv")])
