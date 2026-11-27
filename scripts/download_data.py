"""Скачать датасет объявлений с Hugging Face: python -m scripts.download_data
Файл ~400 МБ. Если скачивание не идёт — скачайте CSV вручную со страницы датасета
и положите в data/raw/russia_real_estate.csv
"""
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "daniilak/Russia_Real_Estate_2018_2021"
TARGET = Path("data/raw/russia_real_estate.csv")

if __name__ == "__main__":
    info = HfApi().dataset_info(REPO, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    print("Файлы в репозитории:")
    for name, size in files:
        print(f"  {name}  {size / 1e6:.1f} МБ")

    csvs = [f for f in files if f[0].lower().endswith(".csv")]
    if not csvs:
        raise SystemExit("CSV-файлов нет. Скачайте данные вручную (см. docstring).")
    name = max(csvs, key=lambda f: f[1])[0]
    print(f"Скачиваю {name} ...")
    src = hf_hub_download(REPO, name, repo_type="dataset")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, TARGET)
    print(f"Готово: {TARGET}")
