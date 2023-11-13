import pandas as pd
import pytest


@pytest.fixture(scope="session")
def roi_crop_csv(tmp_path_factory):
    csv_path = tmp_path_factory.mktemp("data") / "roi.csv"
    data = {
        "SOPInstanceUID": ["1.2.3", "1.2.3", "4.5.6"],
        "x1": [0, 0, 10],
        "y1": [0, 0, 10],
        "x2": [5, 10, 15],
        "y2": [5, 10, 15],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path
