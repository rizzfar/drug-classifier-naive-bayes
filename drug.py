import os
import csv

data = []
with open('drug200.csv', mode='r') as file:
    reader = csv.reader(file)
    i = 0
    for row in reader:
        if i > 0:
            if len(row) == 5:
                data.append({
                    "Age": row[0].strip().upper(),
                    "BP": row[1].strip().upper(),
                    "Cholesterol": row[2].strip().upper(),
                    "Na_to_K": row[3].strip().upper(),
                    "ClassDrug": row[4].strip()
                })
        i += 1

laplace_alpha = 0.5

# Fungsi Ambil Nilai Unik
def get_unique_values(field):
    return list(set([row[field] for row in data]))

# Hitung Probabilitas Prior
def calculate_prior():
    class_counts = {}
    for row in data:
        cls = row["ClassDrug"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    total = len(data)
    return {
        cls: count / total
        for cls, count in class_counts.items()
    }

# Hitung Likelihood
def calculate_likelihood(field, value, target_class):
    match = 0
    class_total = 0
    for row in data:
        if row["ClassDrug"] == target_class:
            class_total += 1
            if row[field] == value:
                match += 1
    feature_values = get_unique_values(field)
    return (match + laplace_alpha) / (class_total + laplace_alpha * len(feature_values))

# Input dari User
def get_user_input():
    os.system("cls" if os.name == "nt" else "clear")
    print("=== INPUT DATA UJI ===")
    age = input("Masukkan Age (YOUNG / ADULT / OLD): ").upper()
    bp = input("Masukkan BP (HIGH / NORMAL / LOW): ").upper()
    chol = input("Masukkan Cholesterol (HIGH / NORMAL): ").upper()
    na_to_k = input("Masukkan Na_to_K (HIGH / NORMAL / LOW): ").upper()
    return age, bp, chol, na_to_k

# Fungsi Prediksi
def predict(age, bp, chol, na_to_k):
    priors = calculate_prior()
    probs = {}
    for cls in priors:
        p = priors[cls]
        p *= calculate_likelihood("Age", age, cls)
        p *= calculate_likelihood("BP", bp, cls)
        p *= calculate_likelihood("Cholesterol", chol, cls)
        p *= calculate_likelihood("Na_to_K", na_to_k, cls)
        probs[cls] = p
    total_prob = sum(probs.values())
    for cls in probs:
        probs[cls] = probs[cls] / total_prob if total_prob > 0 else 0
    return max(probs, key=probs.get), probs

# MAIN PROGRAM
while True:
    age, bp, chol, na_to_k = get_user_input()
    pred_class, probs = predict(age, bp, chol, na_to_k)

    print("\n=== HASIL PREDIKSI ===")
    print(f"Input: Age={age}, BP={bp}, Cholesterol={chol}, Na_to_K={na_to_k}")
    print("\nProbabilitas Tiap Kelas:")
    for cls, prob in probs.items():
        print(f"- {cls}: {round(prob * 100, 2)}%")
    print(f"\n>>> Kesimpulan: Obat yang diprediksi = {pred_class}")

    ulang = input("\nIngin menguji data lain? (y/n): ").lower()
    if ulang != "y":
        print("Terima kasih. Program selesai.")
        break
