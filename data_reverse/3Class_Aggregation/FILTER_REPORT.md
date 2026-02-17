# Aggregation triads filtering + refill (strict)

| category | original | kept | removed | added | final |
|---|---:|---:|---:|---:|---:|
| animal | 251 | 251 | 0 | 0 | 251 |
| item | 382 | 382 | 0 | 0 | 382 |
| organization | 178 | 165 | 13 | 13 | 178 |
| people | 333 | 235 | 98 | 98 | 333 |
| system | 23 | 23 | 0 | 0 | 23 |
| transportation | 33 | 30 | 3 | 3 | 33 |


Files:
- Final triples: `<category>/data.txt`
- Removed list: `<category>/filtered_out.tsv`
- Added list: `<category>/added.tsv`

Constraint: Added (b,c) pairs are unique across all added entries.
