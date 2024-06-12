# ROP Stage
[官中版](./说明.md)

This is the official implementation of the ROP stage section for `ROP-Marker: an evidence-oriented AI assistant for ROP diagnosis`. This method mainly involves sampling ROP lesions and analyzing different sampling results.

`cleansing.py` is used to analyze ridge annotations and staging annotations, and to sample images. For each image, two fields, `stage_sentence_stagelist` and `stage_sentence_path`, will be generated. These fields represent the patch level labels of all patches of this image and the path to the folder storing all patches, respectively. In fact, `stage_sentence_stagelist` is a temporarily deprecated field. In `util.dataset`, patch level labels are actually distinguished by the stored file names (specifically refer to the `CustomDataset` class in that file).

`train.py` and `test.py` are the training and testing files, respectively.

