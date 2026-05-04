# PMI Preparation Summary

- Raw archive: `data/raw/pmi/medication_images.zip`
- README: `data/raw/pmi/image_dataset_readme.txt`
- SHA1: `df15ceb6aa039d71d23774258e28acc89bd31b91`
- Extracted image files: 13955
- Prepared root: `data/prepared/pmi_pills`
- Cost matrix SHA256: `af92dbd367c299056f5f41dab4f3b4330a6171b856e8822bac0357fdfe4289d2`
- Cared classes: `50111-0434`, `53489-0156`, `53746-0544`, `68382-0227`

| Variant | Split | Rows | Classes | Min/Class | Max/Class |
|---|---|---:|---:|---:|---:|
| natural | train | 8393 | 20 | 97 | 602 |
| natural | validation | 1388 | 20 | 16 | 100 |
| natural | calibration | 1388 | 20 | 15 | 100 |
| natural | test | 2786 | 20 | 32 | 200 |
| balanced | train | 1940 | 20 | 97 | 97 |
| balanced | validation | 320 | 20 | 16 | 16 |
| balanced | calibration | 300 | 20 | 15 | 15 |
| balanced | test | 640 | 20 | 32 | 32 |
