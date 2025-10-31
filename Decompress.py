#!/usr/bin/env python3
import argparse, struct
import numpy as np

def cluster_decompress(input_path, output_path):
    with open(input_path, "rb") as f:
        # Odczytaj nagłówek BMP
        header = np.frombuffer(f.read(54), dtype=np.uint8)

        cluster_dict = {}

        # Odczyt klastrów
        while True:
            id_byte = f.read(1)
            if not id_byte:
                raise EOFError("Nie znaleziono znacznika końca klastrów.")
            cluster_id = id_byte[0]
            if cluster_id == 0xFF:
                # to znacznik końca
                f.read(32)
                break
            size_bytes = f.read(4)
            (cluster_size,) = struct.unpack("<I", size_bytes)
            data = f.read(cluster_size)
            cluster_dict[cluster_id] = np.frombuffer(data, dtype=np.uint8)

        # Odczytaj listę ID segmentów
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        print(f"[+] Wczytano {len(cluster_dict)} klastrów i {len(labels)} segmentów")

    # Odtwórz dane BMP (bez nagłówka)
    reconstructed = []
    for lbl in labels:
        reconstructed.append(cluster_dict[lbl])
    reconstructed = np.concatenate(reconstructed)

    # Połącz nagłówek + dane
    full_data = np.concatenate([header, reconstructed])

    # Zapisz jako BMP
    full_data.tofile(output_path)
    print(f"[✓] Zapisano odtworzony plik: {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dekompresja pliku klastrowego BMP z zachowanym nagłówkiem")
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()
    cluster_decompress(args.input, args.output)

