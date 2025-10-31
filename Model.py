import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import entropy

# ====== 1. Wczytanie pliku i zamiana na bity ======
file_path = input("Podaj nazwe pliku ")

with open(file_path, "rb") as f:
    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

bits = np.unpackbits(file_bytes)
print(f"Liczba bitów: {len(bits):,}")

# ====== 2. Ustal parametry ======
segment_size_bits =int(len(bits)/10240)    # 1 KB segment = 8 milionów bitów
n_clusters = 253                 # liczba grup im mniejsza, tym wieksza kompresja

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

print("\n=== CIĄGI BITÓW DLA KAŻDEGO KLASTRA ===")

for cluster_id in np.unique(labels):
    # znajdź segmenty należące do danego klastra
    segment_indices = np.where(labels == cluster_id)[0]
    
    print(f"\n--- Klastr {cluster_id} ---")
    print(f"Liczba segmentów: {len(segment_indices)}")

    # połącz wszystkie bity z segmentów tego klastra
    cluster_bits = np.concatenate([
        bits[i*segment_size_bits : (i+1)*segment_size_bits]
        for i in segment_indices
    ])

    # wypisz pierwszy fragment (bo może być ogromny)
    preview = ''.join(map(str, cluster_bits[:128]))  # 128 pierwszych bitów
    print(f"Przykład bitów (pierwsze 128): {preview}")
    print(f"Łączna liczba bitów w klastrze: {len(cluster_bits):,}\n")

with open("mapa_klastrow.bin","wb") as f:
	for cluster_id in np.unique(labels):
		segment_indices = np.where(labels == cluster_id)[0]
		cluster_bits = np.concatenate([bits[i*segment_size_bits : (i+1)*segment_size_bits] for i in segment_indices])
		#identyfikator klastru w 8bitach		
		f.write(struct.pack("B",cluster_id))
		#ilosc bitów w klastrze
		f.write(struct.pack("<I",len(cluster_bits)))
		cluster_bytes = np.packbits(cluster_bits)
#		print(len(cluster_bits), "dlugosc bitow")
		#faktyczne bity		
		f.write(cluster_bytes)
	f.write(b"xFF"*5)
with open("mapa_klastrow.bin","ab") as f:
	for cluster_id in labels:
	#id klastrow w odpowiedniej kolejnosci
        	f.write(struct.pack("B", cluster_id))
