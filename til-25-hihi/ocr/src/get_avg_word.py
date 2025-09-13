import os

def get_text_stats_from_folder(folder_path: str):
    word_counts = []
    char_counts = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                words = text.strip().split()
                word_counts.append(len(words))
                char_counts.append(len(text))

    if not word_counts:
        return {
            "sample_count": 0,
            "word_stats": {"max": 0, "min": 0, "avg": 0.0},
            "char_stats": {"max": 0, "min": 0, "avg": 0.0}
        }

    return {
        "sample_count": len(word_counts),
        "word_stats": {
            "max": max(word_counts),
            "min": min(word_counts),
            "avg": sum(word_counts) / len(word_counts)
        },
        "char_stats": {
            "max": max(char_counts),
            "min": min(char_counts),
            "avg": sum(char_counts) / len(char_counts)
        }
    }

# Example usage
folder = "/home/jupyter/advanced/ocr"
stats = get_text_stats_from_folder(folder)

print(f"Total Samples: {stats['sample_count']}")
print(f"Word Count - Max: {stats['word_stats']['max']}, Min: {stats['word_stats']['min']}, Avg: {stats['word_stats']['avg']:.2f}")
print(f"Char Count - Max: {stats['char_stats']['max']}, Min: {stats['char_stats']['min']}, Avg: {stats['char_stats']['avg']:.2f}")
