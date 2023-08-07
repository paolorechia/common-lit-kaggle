import polars as pl

block_ranges = [(-5, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (4, 10)]


def create_data_blocks(input_data: pl.DataFrame):
    data_blocks: dict[str, list[pl.DataFrame]] = {"content": [], "wording": []}
    for attr in ["content", "wording"]:
        for block_range in block_ranges:
            block_data = input_data.select(
                input_data.filter(
                    pl.col(attr) > block_range[0] and pl.col(attr) < block_range[1]
                )
            )
            data_blocks[attr].append(block_data)
    return data_blocks


def data_blocks_generator(input_data: pl.DataFrame):
    for attr in ["content", "wording"]:
        for block_range in block_ranges:
            block_data = input_data.select(
                input_data.filter(pl.col(attr) > block_range[0]).filter(
                    pl.col(attr) < block_range[1]
                )
            )
            yield block_data
