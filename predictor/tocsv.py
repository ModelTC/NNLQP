import re
import sys
import numpy as np

fname = sys.argv[1]
fold = int(sys.argv[2])
if len(sys.argv) > 3:
    out_path = "test_{}.csv".format(sys.argv[3])
else:
    out_path = "test.csv"
mapes = []
erbds = []

with open(fname) as f:
    for line in f.readlines():
        if "* MAPE" in line:
            mapes.append(float(line.split(' ')[-1]))
        if "* ErrorBound" in line:
            line = line.replace(']', '').rstrip()
            items = re.split(r'[ \[]', line)
            items = list(filter(lambda x: x!="", items))
            errs = [float(x) for x in items[-1:]]
            erbds.append(errs[0])

mapes = np.array(mapes).reshape(-1, fold)
erbds = np.array(erbds).reshape(-1, fold)
assert mapes.shape[0] == erbds.shape[0]
cnt = mapes.shape[0]

with open(out_path, "w") as f:
    for i in range(cnt):

        s1 = ",".join(["{:.4f}".format(round(x, 4)) for x in mapes[i]])
        s2 = ",".join(["{:.4f}".format(round(x, 4)) for x in erbds[i]])
        m1 = "{:.4f}".format(round(mapes[i].mean(), 4))
        m2 = "{:.4f}".format(round(erbds[i].mean(), 4))

        f.write("{},{},{},{}\n".format(s1, m1, s2, m2))
        print("[{}] mape: {} [{}] erbd: {} [{}]".format(i + 1, s1, m1, s2, m2))

    avg_mape = round(mapes.reshape(-1).mean(), 4)
    avg_erbd = round(erbds.reshape(-1).mean(), 4)

    f.write("," * (fold - 1))
    f.write("{:.4f}".format(avg_mape))
    f.write("," * (fold + 1))
    f.write("{:.4f}".format(avg_erbd))

    print("AVG MAPE: [{:.4f}]".format(avg_mape))
    print("AVG ERBD: [{:.4f}]".format(avg_erbd))
