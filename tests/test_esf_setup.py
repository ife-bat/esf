# General tests for the esf module


def test_esf():
    # Simple test to check if the module can be imported and that
    # the test runner outputs the print statements and logging messages.

    import logging

    print("Print msg")
    # This will not be shown unless logging level is set to DEBUG
    logging.debug("Debug msg")
    # This will not be shown unless logging level is set to DEBUG or INFO:
    logging.info("Info msg")
    # These should be shown:
    logging.error("Error msg")
    logging.warning("Warning msg")
    logging.critical("Critical msg")
    logging.exception("Exception msg")


def test_load_test_data(shared_datadir):
    # This test uses the shared_datadir fixture provided by pytest-datadir plugin.
    # If missing, install with: pip install pytest-datadir

    print(f"Shared datadir tmp location:\n {shared_datadir}")

    assert shared_datadir.is_dir()
    assert len(list(shared_datadir.glob("*"))) > 0

    print("Listing files in shared_datadir:")
    for i, n in enumerate(shared_datadir.glob("*")):
        print(f" {i:03} {n.name}")
