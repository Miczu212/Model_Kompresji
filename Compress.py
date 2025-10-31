#!/usr/bin/env python3
import argparse, struct
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def cluster_compress(input_path, output_path, chunk_size=64, n_clusters=128):
    data = np.fromfile(input_path, dtype=np.uint8)
    print(f"[+] Wczytano {len(data)} bajtów z {input_path}")

    # Zachowaj nagłówek BMP (54 bajty)
    header = data[:54]
    body = data[54:]

    # Dopasowanie długości
    pad = (-len(body)) % chunk_size
    if pad:
        body = np.concatenate([body, np.zeros(pad, dtype=np.uint8)])
    chunks = body.reshape((-1, chunk_size))
    print(f"[+] Segmenty: {len(chunks)} x {chunk_size}")

    # Uczenie KMeans
    print(f"[+] KMeans (n_clusters={n_clusters}) ...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=42)
    labels = kmeans.fit_predict(chunks)
    centers = np.uint8(np.clip(kmeans.cluster_centers_, 0, 255))

    # Zapis do pliku
    print("[+] Zapis pliku w formacie użytkownika ...")
    with open(output_path, "wb") as f:
        # najpierw nagłówek BMP
        f.write(header)

        # zapis klastrów
        for cluster_id in range(n_clusters):
            center = centers[cluster_id].tobytes()
            f.write(struct.pack("B", cluster_id))           # 8 bit ID
            f.write(struct.pack("<I", len(center)))         # 32 bit rozmiar
            f.write(center)                                 # dane

        # znacznik końca
        f.write(b'\xFF' + b'\xFF' * 32)

        # lista ID segmentów
        f.write(labels.astype(np.uint8).tobytes())

    print(f"[✓] Zapisano {output_path}")
    print("[!] BMP nagłówek zachowany bez zmian (stratna kompresja danych).")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kompresja BMP do własnego formatu z zachowaniem nagłówka BMP")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--n-clusters", type=int, default=128)
    args = p.parse_args()

    cluster_compress(args.input, args.output, args.chunk_size, args.n_clusters)

