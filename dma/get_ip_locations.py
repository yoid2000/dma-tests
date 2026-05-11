import csv
import gzip

locations = set()

with gzip.open("dbip-city-lite-2026-05.csv.gz", "rt", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        ip_start, ip_end, continent, country, stateprov, city, lat, lon = row
        locations.add((ip_start, ip_end, continent, country, stateprov, city, lat, lon))

with open("distinct_locations.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["ip_start", "ip_end", "continent", "country", "stateprov", "city", "latitude", "longitude"]
    )
    writer.writerows(sorted(locations))
