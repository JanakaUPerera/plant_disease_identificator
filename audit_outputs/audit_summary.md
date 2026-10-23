# Pre-proposal data audit

Mapping file: `D:\MSc\Projects\module_07_artificial_nurel_network\cw\plant_disease_identificator\class_mapping.json` · tiers audited: candidate, core · min count per evaluation set: 10 · near-duplicate rule: pHash ≤ 6 and dHash ≤ 8 (64-bit hashes)

## Datasets found

- **PlantVillage**: 54,305 images in total; class folders: `data\plantvillage\raw\color`
- **PlantDoc**: 2,572 images in total; train/test folders under: `data\plantdoc`
- **PlantWild v1**: 18,542 images in total; class folders: `data\plantwild\v1\images`; v1 (89 classes, includes healthy classes); official split file: `data\plantwild\v1\trainval.txt`
- **PlantWild v2**: 11,488 images in total; class folders: `data\plantwild\v2`; v2 (disease-only classes: healthy classes cannot be tested on PlantWild); official split file: not found (split reported as unknown)

## Headline numbers

- Core classes: 20; core classes meeting the minimum count in every evaluation set: 8
- PlantDoc test images per core class: median 9, min 4 (corn_gray_leaf_spot)
- PlantWild external-test images per core class (v2, or v1 where v2 has no folder): median 134, min 63 (tomato_mosaic_virus)
- PlantWild v2 images per core class: median 120, min 0 (apple_healthy)
- PlantWild v1 images per core class: median 233, min 117 (pepper_bacterial_spot)
- PlantDoc train vs PlantDoc test (split leakage): 15 duplicate pairs (14 exact, 10 with conflicting labels); images affected: 15 of the first set (0.64%), 14 of the second (5.93%)
- PlantWild v2 vs PlantDoc (external-test independence): 244 duplicate pairs (178 exact, 55 with conflicting labels); images affected: 220 of the first set (8.1%), 231 of the second (8.98%)
- PlantWild v1 vs PlantDoc (external-test independence): 416 duplicate pairs (282 exact, 109 with conflicting labels); images affected: 376 of the first set (4.9%), 353 of the second (13.72%)
- PlantVillage vs PlantDoc: 0 duplicate pairs (0 exact, 0 with conflicting labels); images affected: 0 of the first set (0.0%), 0 of the second (0.0%)
- PlantVillage vs PlantWild v2: 30 duplicate pairs (5 exact, 2 with conflicting labels); images affected: 30 of the first set (0.06%), 30 of the second (1.1%)
- PlantVillage vs PlantWild v1: 48 duplicate pairs (6 exact, 3 with conflicting labels); images affected: 48 of the first set (0.1%), 48 of the second (0.63%)
- PlantWild v1 vs PlantWild v2 (release overlap): 2949 duplicate pairs (2839 exact, 222 with conflicting labels); images affected: 2793 of the first set (36.38%), 2645 of the second (97.42%)
- Within PlantDoc: 72 duplicate pairs (47 exact, 35 with conflicting labels); images affected: 138 (5.37%)
- Within PlantWild v2: 83 duplicate pairs (50 exact, 46 with conflicting labels); images affected: 161 (5.93%)
- Within PlantWild v1: 353 duplicate pairs (224 exact, 236 with conflicting labels); images affected: 593 (7.72%)
- PlantWild v2 has no folder for 12 mapped classes (apple_healthy, blueberry_healthy, cherry_healthy, corn_healthy, grape_healthy, peach_healthy, pepper_healthy, potato_healthy, raspberry_healthy, soybean_healthy, strawberry_healthy, tomato_healthy); their external-test images come from PlantWild v1.

## Images per class

PD_train is the in-the-wild validation set and PD_test the same-source test set in the proposed design. PW1_* and PW2_* are the two PlantWild releases (PW1 splits from the official v1 trainval.txt); PW_external is the external test set (v2, or v1 for classes v2 lacks) and PW_external_from its source.

| class_id | crop | tier | approximate_mapping | PV_all | PD_train | PD_test | PW1_train | PW1_val | PW1_test | PW1_total | PW2_total | PW_external | PW_external_from | below_min_10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apple_scab | apple | core | False | 630 | 83 | 10 | 205 | 29 | 58 | 292 | 258 | 258 | v2 |  |
| apple_cedar_rust | apple | core | True | 275 | 78 | 10 | 217 | 30 | 61 | 308 | 139 | 139 | v2 |  |
| apple_healthy | apple | core | False | 1645 | 82 | 9 | 312 | 44 | 88 | 444 | 0 | 444 | v1 | field-test=9 |
| pepper_bacterial_spot | bell_pepper | core | True | 997 | 62 | 9 | 83 | 11 | 23 | 117 | 76 | 76 | v2 | field-test=9 |
| pepper_healthy | bell_pepper | core | False | 1478 | 53 | 8 | 168 | 24 | 48 | 240 | 0 | 240 | v1 | field-test=8 |
| corn_gray_leaf_spot | corn | core | False | 513 | 64 | 4 | 152 | 21 | 43 | 216 | 107 | 107 | v2 | field-test=4 |
| corn_northern_leaf_blight | corn | core | True | 985 | 179 | 12 | 158 | 22 | 44 | 224 | 130 | 130 | v2 |  |
| corn_common_rust | corn | core | False | 1192 | 106 | 10 | 167 | 23 | 47 | 237 | 177 | 177 | v2 |  |
| grape_black_rot | grape | core | False | 1180 | 56 | 8 | 161 | 23 | 45 | 229 | 122 | 122 | v2 | field-test=8 |
| grape_healthy | grape | core | False | 423 | 57 | 12 | 141 | 20 | 40 | 201 | 0 | 201 | v1 |  |
| potato_early_blight | potato | core | False | 1000 | 108 | 8 | 160 | 22 | 45 | 227 | 126 | 126 | v2 | field-test=8 |
| potato_late_blight | potato | core | False | 1000 | 97 | 8 | 168 | 24 | 48 | 240 | 117 | 117 | v2 | field-test=8 |
| tomato_bacterial_spot | tomato | core | False | 2127 | 101 | 9 | 196 | 28 | 56 | 280 | 110 | 110 | v2 | field-test=9 |
| tomato_early_blight | tomato | core | False | 1000 | 79 | 9 | 243 | 34 | 69 | 346 | 187 | 187 | v2 | field-test=9 |
| tomato_late_blight | tomato | core | False | 1909 | 101 | 10 | 207 | 29 | 59 | 295 | 163 | 163 | v2 |  |
| tomato_leaf_mold | tomato | core | False | 952 | 85 | 6 | 168 | 24 | 47 | 239 | 156 | 156 | v2 | field-test=6 |
| tomato_septoria_leaf_spot | tomato | core | False | 1771 | 140 | 11 | 154 | 22 | 44 | 220 | 130 | 130 | v2 |  |
| tomato_mosaic_virus | tomato | core | False | 373 | 44 | 10 | 133 | 19 | 37 | 189 | 63 | 63 | v2 |  |
| tomato_yellow_leaf_curl | tomato | core | False | 5357 | 70 | 6 | 120 | 17 | 34 | 171 | 93 | 93 | v2 | field-test=6 |
| tomato_healthy | tomato | core | False | 1591 | 55 | 8 | 159 | 22 | 45 | 226 | 0 | 226 | v1 | field-test=8 |
| apple_black_rot | apple | candidate | False | 621 | 0 | 0 | 122 | 17 | 34 | 173 | 83 | 83 | v2 |  |
| corn_healthy | corn | candidate | False | 1162 | 0 | 0 | 110 | 15 | 31 | 156 | 0 | 156 | v1 |  |
| potato_healthy | potato | candidate | False | 152 | 0 | 0 | 175 | 25 | 49 | 249 | 0 | 249 | v1 |  |
| cherry_powdery_mildew | cherry | candidate | False | 1052 | 0 | 0 | 81 | 11 | 22 | 114 | 33 | 33 | v2 |  |
| strawberry_leaf_scorch | strawberry | candidate | False | 1109 | 0 | 0 | 54 | 7 | 15 | 76 | 39 | 39 | v2 |  |
| grape_leaf_blight | grape | candidate | True | 1076 | 0 | 0 | 75 | 10 | 21 | 106 | 91 | 91 | v2 |  |
| citrus_greening | citrus | candidate | True | 5507 | 0 | 0 | 167 | 23 | 47 | 237 | 133 | 133 | v2 |  |
| tomato_spider_mites | tomato | candidate | True | 1676 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | field-val=2; field-test=0 |
| cherry_healthy | cherry | candidate | False | 854 | 47 | 10 | 201 | 28 | 57 | 286 | 0 | 286 | v1 |  |
| strawberry_healthy | strawberry | candidate | False | 456 | 88 | 8 | 140 | 20 | 39 | 199 | 0 | 199 | v1 | field-test=8 |
| peach_healthy | peach | candidate | False | 360 | 102 | 9 | 111 | 15 | 31 | 157 | 0 | 157 | v1 | field-test=9 |
| blueberry_healthy | blueberry | candidate | False | 1502 | 104 | 11 | 197 | 28 | 56 | 281 | 0 | 281 | v1 |  |
| raspberry_healthy | raspberry | candidate | False | 371 | 112 | 7 | 126 | 18 | 36 | 180 | 0 | 180 | v1 | field-test=7 |
| soybean_healthy | soybean | candidate | False | 5090 | 57 | 8 | 170 | 24 | 48 | 242 | 0 | 242 | v1 | field-test=8 |
| squash_powdery_mildew | squash | candidate | False | 1835 | 124 | 6 | 197 | 28 | 56 | 281 | 182 | 182 | v2 | field-test=6 |

## Duplicate search

Near-duplicates are candidates: open `duplicate_pairs/*.jpg` and confirm them by eye before removing. In PlantWild v1 vs v2, same-class pairs are expected (v2 reuses v1 images); pairs with conflicting labels are the ones to check.

| comparison | images_a | images_b | pairs | exact_pairs | label_conflict_pairs | affected_images_a | affected_pct_a | affected_images_b | affected_pct_b |
|---|---|---|---|---|---|---|---|---|---|
| PlantDoc train vs PlantDoc test (split leakage) | 2336 | 236 | 15 | 14 | 10 | 15 | 0.64 | 14 | 5.93 |
| PlantWild v2 vs PlantDoc (external-test independence) | 2715 | 2572 | 244 | 178 | 55 | 220 | 8.1 | 231 | 8.98 |
| PlantWild v1 vs PlantDoc (external-test independence) | 7678 | 2572 | 416 | 282 | 109 | 376 | 4.9 | 353 | 13.72 |
| PlantVillage vs PlantDoc | 49221 | 2572 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0.0 |
| PlantVillage vs PlantWild v2 | 49221 | 2715 | 30 | 5 | 2 | 30 | 0.06 | 30 | 1.1 |
| PlantVillage vs PlantWild v1 | 49221 | 7678 | 48 | 6 | 3 | 48 | 0.1 | 48 | 0.63 |
| PlantWild v1 vs PlantWild v2 (release overlap) | 7678 | 2715 | 2949 | 2839 | 222 | 2793 | 36.38 | 2645 | 97.42 |
| Within PlantDoc | 2572 |  | 72 | 47 | 35 | 138 | 5.37 |  |  |
| Within PlantWild v2 | 2715 |  | 83 | 50 | 46 | 161 | 5.93 |  |  |
| Within PlantWild v1 | 7678 |  | 353 | 224 | 236 | 593 | 7.72 |  |  |

## Image quality (mapped classes)

| dataset | images | unreadable | min_side_lt_100px | min_side_lt_224px | greyscale_or_palette | median_width | median_height |
|---|---|---|---|---|---|---|---|
| PlantDoc | 2572 | 0 | 3 | 111 | 1 | 800.0 | 667.5 |
| PlantVillage | 49221 | 0 | 0 | 0 | 0 | 256.0 | 256.0 |
| PlantWild v1 | 7678 | 0 | 45 | 595 | 0 | 660.0 | 514.0 |
| PlantWild v2 | 2715 | 0 | 7 | 232 | 0 | 640.0 | 500.0 |

## Folders not in the mapping

151 folders (24721 images) are not in the audited tiers; see `unmapped_folders.csv`. PlantWild classes outside the label space are useful later as out-of-distribution test inputs for the web application.
