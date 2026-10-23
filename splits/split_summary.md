# Fixed data splits

Seed 42 · PlantVillage fractions 0.70,0.15,0.15 · label space: 20 classes (tiers: core) · healthy wild test: plantwild_v1

- PlantWild disease classes from v2: `data\plantwild\v2`
- PlantWild v1 used only for: apple_healthy, grape_healthy, pepper_healthy, tomato_healthy (`data\plantwild\v1\images`)
- PlantVillage leaf grouping from `data\plantvillage\leaf-map.json`: 17,937 of 26,397 images matched to 4,004 leaves; 4,001 leaves have more than one photo (up to 17); unmatched images form groups of their own
- Duplicate rule: pHash ≤ 6 and dHash ≤ 8; PlantVillage copies grouped at pHash ≤ 2 and dHash ≤ 2; largest PlantVillage group (same leaf or copies): 17 images
- Pairs excluded after visual check (--not-duplicates): 0

## Images per class and split

| class_id | train | val_lab | test_lab | val_wild | test_wild |
|---|---|---|---|---|---|
| apple_scab | 440 | 95 | 95 | 91 | 249 |
| apple_cedar_rust | 192 | 43 | 40 | 86 | 132 |
| apple_healthy | 1151 | 247 | 247 | 89 | 430 |
| pepper_bacterial_spot | 699 | 151 | 147 | 67 | 57 |
| pepper_healthy | 1034 | 222 | 222 | 61 | 234 |
| corn_gray_leaf_spot | 359 | 77 | 77 | 64 | 88 |
| corn_northern_leaf_blight | 689 | 148 | 148 | 175 | 115 |
| corn_common_rust | 834 | 179 | 179 | 115 | 155 |
| grape_black_rot | 828 | 176 | 176 | 63 | 112 |
| grape_healthy | 296 | 64 | 63 | 69 | 196 |
| potato_early_blight | 700 | 152 | 148 | 88 | 91 |
| potato_late_blight | 700 | 151 | 148 | 91 | 94 |
| tomato_bacterial_spot | 1488 | 320 | 319 | 99 | 97 |
| tomato_early_blight | 700 | 151 | 149 | 73 | 143 |
| tomato_late_blight | 1336 | 287 | 286 | 98 | 148 |
| tomato_leaf_mold | 664 | 144 | 144 | 88 | 141 |
| tomato_septoria_leaf_spot | 1240 | 266 | 265 | 137 | 90 |
| tomato_mosaic_virus | 261 | 56 | 56 | 51 | 57 |
| tomato_yellow_leaf_curl | 3750 | 804 | 803 | 72 | 76 |
| tomato_healthy | 1114 | 239 | 238 | 62 | 211 |
| TOTAL | 18475 | 3972 | 3950 | 1739 | 2916 |

## Images removed by each cleaning rule

| rule | PlantDoc | PlantVillage | PlantWild |
|---|---|---|---|
| R1 | 3 | 0 | 6 |
| R2 | 106 | 1 | 128 |
| R3 | 0 | 0 | 179 |
| R4 | 29 | 0 | 36 |

R1 unreadable or too small · R5 rejected in the manual label review, with all its copies (applied after R1, before R2) · R2 copies with conflicting labels (all copies removed) · R3 copy of an image already in an earlier set · R4 repeated copy inside one set.

## Leakage check: duplicate pairs between final splits (should all be 0)

| split_a | split_b | duplicate_pairs |
|---|---|---|
| train | val_lab | 0 |
| train | test_lab | 0 |
| train | val_wild | 0 |
| train | test_wild | 0 |
| val_lab | test_lab | 0 |
| val_lab | val_wild | 0 |
| val_lab | test_wild | 0 |
| test_lab | val_wild | 0 |
| test_lab | test_wild | 0 |
| val_wild | test_wild | 0 |

**Same-leaf check:** PlantVillage leaves with photos in more than one split: 0 (should be 0).

## Split fingerprints (SHA-256 of dataset|relative path|label, first 16 hex digits; same on every OS)

- `train.csv`: 0aaff7ac5e582abf
- `val_lab.csv`: 6a77ada2f201fabe
- `test_lab.csv`: 6c99ca503ff3e4e0
- `val_wild.csv`: b569b2be896440cb
- `test_wild.csv`: 2a9359f5ed907f1e
