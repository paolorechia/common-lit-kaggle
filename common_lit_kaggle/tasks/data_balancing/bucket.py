from dataclasses import dataclass

import polars as pl


@dataclass
class BucketResult:
    content_histogram: pl.DataFrame
    content_min_label_size: int
    content_max_label_size: int
    wording_histogram: pl.DataFrame
    wording_min_label_size: int
    wording_max_label_size: int


def bucket_data(input_data: pl.DataFrame) -> BucketResult:
    # Bin counts manually decided
    content_histogram = (
        input_data.select(pl.col("content"))
        .to_series()
        .hist(bin_count=8)
        .filter(pl.col("content_count") > 0)
    )
    min_content_label_size = content_histogram.select(
        pl.col("content_count").min()
    ).to_numpy()[0][0]
    max_content_label_size = content_histogram.select(
        pl.col("content_count").max()
    ).to_numpy()[0][0]

    wording_histogram = (
        input_data.select(pl.col("wording"))
        .to_series()
        .hist(bin_count=6)
        .filter(pl.col("wording_count") > 0)
    )
    min_wording_label_size = wording_histogram.select(
        pl.col("wording_count").min()
    ).to_numpy()[0][0]
    max_wording_label_size = wording_histogram.select(
        pl.col("wording_count").max()
    ).to_numpy()[0][0]
    result = BucketResult(
        content_histogram=content_histogram,
        content_min_label_size=min_content_label_size,
        content_max_label_size=max_content_label_size,
        wording_histogram=wording_histogram,
        wording_min_label_size=min_wording_label_size,
        wording_max_label_size=max_wording_label_size,
    )
    return result
