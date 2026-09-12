"""Self-consistency detector adaptation using five shared answer support votes."""

from .sample_consistency import main_for


def main(argv=None):
    main_for("self_consistency", argv)


if __name__ == "__main__":
    main()
