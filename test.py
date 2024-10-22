from torch.utils.tensorboard.writer import SummaryWriter
import keyword
import torch

writer = SummaryWriter(flush_secs=10)
meta = []
while len(meta) < 100:
    meta = meta + list(keyword.kwlist)  # get some strings
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = v + str(i)

label_img = torch.rand(100, 3, 10, 32)
for i in range(100):
    label_img[i] *= i / 100.0

writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
# writer.add_embedding(torch.randn(100, 5), label_img=label_img)
# writer.add_embedding(torch.randn(100, 5), metadata=meta)
# writer.close()
writer.flush()
writer.close()

# tensorboard --logdir=runs
