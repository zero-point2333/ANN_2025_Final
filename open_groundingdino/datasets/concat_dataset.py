import bisect
from datasets.coco import CocoDetection
from datasets.odvg import ODVGDataset


class ConcatDataset():
    r"""Dataset as a concatenation of multiple datasets.
    Args:
        datasets (sequence): List of datasets to be concatenated

    datasets: List[Dataset[T_co]]  #存储要拼接的多个数据集的列表
    cumulative_sizes: List[int] #存储到每个数据集末尾为止的累积数据集大小的列表。
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets) -> None:
        super().__init__()
        for dataset in datasets:
            assert isinstance(dataset, (CocoDetection, ODVGDataset))
        self.cumulative_sizes = self.cumsum(self.datasets)
        #使用 cumsum 方法计算每个数据集的累积大小，以便于后续快速索引
    def __len__(self):
        return self.cumulative_sizes[-1]

    #用于检索特定索引 idx 处的数据。
    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx  #如果 idx 为负数，则将其转换为正数索引。
        #使用二分查找（bisect_right）确定 idx 属于哪个子数据集。
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        #计算在该子数据集中的相对索引，并返回相应的数据。
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]