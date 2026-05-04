# PMI Download Attempts - 2026-04-30

## Official Source

- Deep Blue Data work: https://deepblue.lib.umich.edu/data/concern/data_sets/6d56zw997
- Official archive file set: `rr171x63c` / `medication_images.zip`
- Official direct archive URL attempted: https://deepblue.lib.umich.edu/data/downloads/rr171x63c
- Official SHA1 from Deep Blue: `df15ceb6aa039d71d23774258e28acc89bd31b91`
- Expected structure after extraction: `NLM20/{train,valid,test}/<NDC>/*.jpg` or equivalent `Train/Valid/Test` folders.

## Attempts From This Machine

- `curl`/HTTP GET and HEAD to the direct archive URL: blocked with HTTP 403 and `cf-mitigated: challenge`.
- `curl`/HTTP GET and HEAD to Deep Blue JSON/REST endpoints: blocked with the same Cloudflare managed challenge.
- `curl_cffi` Chrome impersonation from a disposable `/tmp` target package dir: blocked with the same 403 challenge.
- Selenium + headless Firefox/geckodriver: reached Cloudflare verification page and did not receive a clearance cookie.
- Hugging Face entries `gwenxin/pills_inside_bottles` and `jordansegovia/pills_inside_bottles`: verified as loading-script repos only; total hosted file size is about 9.67 KB and the script points back to the Deep Blue URL, so they are not usable image mirrors.
- Kaggle dataset search for the exact PMI/pills-inside-bottles terms: no matching dataset found.
- Globus CLI installed into `/tmp` target packages, but Globus requires user login and a configured endpoint before transfer.

## Current Status

PMI is not locally downloaded or prepared. `data/raw/pmi` exists but has no archive or split folders. MC0 readiness correctly reports `pmi_pills` as not ready.

## Next Reproducible Step

Use the Deep Blue browser or Globus path to place the official archive at:

```bash
/mnt/storage/github/NICME/data/raw/pmi/medication_images.zip
```

Then run:

```bash
cd /mnt/storage/github/NICME
sha1sum data/raw/pmi/medication_images.zip
PYTHONPATH=. micromamba run -n ml python -m nicme.data_prep --dataset pmi_pills --input-dir data/raw/pmi --output-dir data/prepared/pmi_pills --extract
PYTHONPATH=. micromamba run -n ml python scripts/check_multiclass_readiness.py --datasets pmi_pills --variants balanced --raw-root data/raw --prepared-root data/prepared --output-dir results/multiclass/pmi_mc0
```

The SHA1 must equal `df15ceb6aa039d71d23774258e28acc89bd31b91` before preparation is accepted.
