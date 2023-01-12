import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser("VALL-E TTS")
    parser.add_argument("text")
    parser.add_argument("output")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()


if __name__ == "__main__":
    main()
