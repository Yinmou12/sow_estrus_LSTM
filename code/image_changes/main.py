from pathlib import Path

from change_dpi import check_and_update_dpi

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "inputs"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

INPUT_FILE = INPUT_DIR / "image-v01-1.png"
OUTPUT_FILE = OUTPUT_DIR / "image-v01-2000dpi.png"

TARGET_DPI = 2000


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_and_update_dpi(str(INPUT_FILE), str(OUTPUT_FILE), target_dpi=TARGET_DPI)


if __name__ == "__main__":
    main()
