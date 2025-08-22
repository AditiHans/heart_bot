import matplotlib.pyplot as plt
import os

def parse_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    em = None
    f1 = None
    for line in lines:
        if "Exact Match" in line:
            em = float(line.strip().split(":")[-1])
        if "F1 Score" in line:
            f1 = float(line.strip().split(":")[-1])
    return em, f1

def main():
    os.makedirs("results", exist_ok=True)  # auto-create results/
    files = {
        "TaskA": "results/taskA_on_A.txt",
        "TaskB": "results/taskB_on_A.txt",
        "TaskB+Replay": "results/taskBcont_on_A.txt"
    }

    em_scores = []
    f1_scores = []

    for label, path in files.items():
        em, f1 = parse_file(path)
        em_scores.append((label, em))
        f1_scores.append((label, f1))

    # Plot F1 forgetting
    labels = [x[0] for x in f1_scores]
    values = [x[1] for x in f1_scores]

    plt.figure(figsize=(7,5))
    plt.bar(labels, values, color=["#4CAF50", "#F44336", "#2196F3"])
    plt.ylabel("F1 Score")
    plt.title("Forgetting Effect on Task A")
    plt.savefig("results/forgetting_f1.png")
    print("Saved plot to results/forgetting_f1.png")

if __name__ == "__main__":
    main()