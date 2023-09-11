import random as rd
import csv

# Col 2 - 1.0 to 4.0 (mean=3)
# Col 3 - 17/23 No(0)
# Col 4 - 0 to 5 (mean=2)
# Col 5 - Mobile, Desktop, Tablet, Laptop (Random)
# Col 6 - Cities Random
# Col 7 - 37/53 Consistent(1)
# Col 8 - Frequent, Normal, Rare
# Col 9 - 100 to 500 (mean=300)
# Col 10 - 31/43 No(0)
# Col 11 - 43/71 No(0)
# Col 12 - 57/96 No(0)
# Col 13 - 5/96 Yes(1)

name = "newgen.csv"
dev_type = ["Desktop", "Laptop", "Mobile", "Tablet"]
cities = ["Chennai", "Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Lucknow", "Patna"]
user_hist = ["normal", "frequent", "rare"]

with open(name, 'w', newline="") as cf:
    csv.writer(cf).writerow(["Transaction_ID", "Interaction_Speed", "Input_Hesitation", "Repeated_Attempts", "Device_Type", "IP_Location", "Typing_Patterns", "User_History", "Session_Length", "Geographical_Anomaly", "Browser_Change", "Uncommon_Service", "Fraud"])
    cf.close

for i in range(100000):
    with open(name, "a", newline="") as cf:
        int_spd = 5.5 - rd.lognormvariate(1.0, 0.175) # Col 2
        io_hes = 0 if rd.random() > 17/23 else 1
        rep_att = int(rd.weibullvariate(1.2,1.4))
        dev = rd.choice(dev_type)
        cit = rd.choice(cities)
        typ_pat = 1 if rd.random() > 37/53 else 0
        use_his = rd.choices(user_hist, weights=(17, 5, 3), k=1)[0]
        ses_len = 10 * int(10*rd.weibullvariate(3,4))
        geo_ano = 0 if rd.random() > 31/43 else 1
        bro_cha = 0 if rd.random() > 43/71 else 1
        unc_ser = 0 if rd.random() > 57/96 else 1
        fraud = 0
        writeobj = csv.writer(cf)
        writeobj.writerow([i+1, int_spd, io_hes, rep_att, dev, cit, typ_pat, use_his, ses_len, geo_ano, bro_cha, unc_ser, fraud])
        cf.close()
