from collections import defaultdict

def parse_raw_profile(filepath):
    samples = defaultdict(int)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # Формат: "func1;func2;func3 123"
            stack_part, count_str = line.rsplit(' ', 1)
            count = int(count_str)
            # Берем только последнюю функцию в стеке (где реально было время)
            leaf_func = stack_part.split(';')[-1]
            samples[leaf_func] += count
    return samples

# Использование:
profile_data = parse_raw_profile('profile.raw')
total = sum(profile_data.values())
print(f"{'Function':<50} | {'Samples':>8} | {'%':>6}")
print("-" * 70)
for func, count in sorted(profile_data.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"{func[:50]:<50} | {count:>8} | {(count/total)*100:>5.1f}%")