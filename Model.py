import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import entropy

# ====== 1. Wczytanie pliku i zamiana na bity ======
file_path = "2019-06-20-Jezioro-Lednickie-010-Pano-1024x663.jpg"

with open(file_path, "rb") as f:
    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

bits = np.unpackbits(file_bytes)
print(f"Liczba bitów: {len(bits):,}")

# ====== 2. Ustal parametry ======
segment_size_bits =int(len(bits)/1000)    # 1 MB segment = 8 milionów bitów
n_clusters = 1000                 # liczba grup

n_segments = len(bits) // segment_size_bits
print(f"Liczba segmentów: {n_segments}")

# ====== 3. Wylicz cechy dla każdego segmentu ======
features = []

for i in range(n_segments):
    seg = bits[i*segment_size_bits : (i+1)*segment_size_bits]

    # podstawowe statystyki:
    mean = np.mean(seg)  # udział jedynek
    var = np.var(seg)
    ones = np.sum(seg)
    zeros = len(seg) - ones

    # histogram długości serii bitów (1/0)
    diff = np.diff(seg)
    runs = np.sum(diff != 0)  # liczba przejść 0→1 lub 1→0

    # entropia
    p = np.bincount(seg) / len(seg)
    H = entropy(p, base=2)

    # wektor cech segmentu
    features.append([mean, var, runs / len(seg), H])

features = np.array(features)
print(f"Macierz cech: {features.shape}")

# ====== 4. Klasteryzacja dużych segmentów ======
model = MiniBatchKMeans(n_clusters=n_clusters, random_state=412, batch_size=100)
model.fit(features)
labels = model.labels_

unique_labels = np.unique(labels)
cluster_names = {label: f"{label}\n" for label in unique_labels}

# Generujemy listę nazw dla każdego segmentu
segment_labels = [cluster_names[label] for label in labels]

# ====== 5. Wyniki ======
for i, label in enumerate(labels):
    print(f"Segment {i}: klaster {label}")
plt.figure(figsize=(12, 5))


# --- (A) Wykres w przestrzeni cech (entropia vs średnia liczba 1)
plt.subplot(1, 2, 1)
plt.scatter(features[:, 0], features[:, 3], c=labels, s=40, cmap='tab20')
plt.xlabel("Średnia liczba bitów '1'")
plt.ylabel("Entropia segmentu")
plt.title("Klastry w przestrzeni cech")
plt.colorbar(label="Numer klastra")

# --- (B) Wykres kolejności segmentów w pliku
plt.subplot(1, 2, 2)
plt.scatter(range(len(labels)), labels, c=labels, cmap='tab20', s=40)
plt.xlabel("Numer segmentu (kolejność w pliku)")
plt.ylabel("Numer klastra")
plt.title("Rozmieszczenie klastrów w pliku")
#plt.tight_layout()
#plt.show()

with open("mapa_klastrow.bin", "wb") as f:
	for i, (label, name) in enumerate(zip(labels, segment_labels)):
		text_bytes = name.encode('utf-8')
		bits = np.unpackbits(np.frombuffer(text_bytes, dtype=np.uint8))
		f.write(bits)
		name="\n"
		text_bytes = name.encode('utf-8')
		bits = np.unpackbits(np.frombuffer(text_bytes, dtype=np.uint8))
		f.write(bits)
with open("mapa_klastrow.bin", "wb") as f:
	text="\n\n"
	text_bytes = text.encode('utf-8')
	bits = np.unpackbits(np.frombuffer(text_bytes, dtype=np.uint8))
	f.write(bits)
with open("mapa_klastrow.bin", "wb") as f:
	for label in unique_labels:
		indices = np.where(labels == label)[0]
		cluster_bits = np.concatenate([bits[i*segment_size_bits:(i+1)*segment_size_bits] for i in indices])
		cluster_bytes = np.packbits(cluster_bits)
		f.write(cluster_bytes)
		text="\n"
		text_bytes = text.encode('utf-8')
		bits = np.unpackbits(np.frombuffer(text_bytes, dtype=np.uint8))
		f.write(bits)
