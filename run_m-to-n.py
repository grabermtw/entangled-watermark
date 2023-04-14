import itertools
import subprocess
import os

datasets = [("mnist", 10), ("fashion", 10), ("speechcmd", 10)]
max_m = 4
max_n = 4

for dataset, num_class in datasets:
    # iterate over the different values of m and n
    for m in range(1,max_m):
        for n in range(1,max_n):
            # skip 1-to-1 watermarking because the original authors already investigated that
            if not (m == n and m == 1):
                # iterate over the different positions of m and n
                m_combos = list(itertools.combinations(range(num_class), m))
                n_combos = list(itertools.combinations(range(num_class), n))
                outfile = os.path.join("results", dataset + "_" + str(m) + "_" + str(n) + ".csv")
                if not os.path.isfile(outfile):
                    os.makedirs(os.path.dirname(outfile), exist_ok=True)
                    with open(outfile, "w+") as f:
                        f.write("Sources,Targets,Victim Accuracy,Victim WM Success Rate,Extracted Accuracy,Extracted WM Success Rate,Baseline Accuracy,Baseline WM Success Rate")
                for m_combo in m_combos:
                    for n_combo in n_combos:
                        # Skip when they have common elements
                        if not (set(m_combo) & set(n_combo)):
                            cmd = "python3 train.py --dataset {0} --source {1} --target {2} --outfile {3}".format(dataset, ' '.join(str(x) for x in m_combo), ' '.join(str(x) for x in n_combo), outfile)
                            print("Running command:", cmd)
                            subprocess.run(cmd, shell=True)
                exit(0)
                    