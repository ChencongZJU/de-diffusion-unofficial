# import torch

# # 创建两个不同形状的张量
# x = torch.tensor([[1, 2, 3]])
# y = torch.tensor([[4, 5, 6], [7, 8, 9]])

# # 调整张量形状使其匹配
# x = x.unsqueeze(0).to(dtype=torch.float)  # 在第0维上增加一个维度
# y = y.unsqueeze(1).to(dtype=torch.float)  # 在第1维上增加一个维度

# # 计算余弦相似度
# similarity = torch.cosine_similarity(x, y, dim=-1)
# print(similarity)
# print(x.shape)
# print(y.shape)

import torch

# 创建两个张量
a = torch.randn(77, 10)
b = torch.randn(49408, 10)

# 调整张量形状使其匹配
a = a.unsqueeze(0)  # 在第0维上增加一个维度
b = b.unsqueeze(1)  # 在第1维上增加一个维度

# 计算余弦相似度
similarity = torch.cosine_similarity(a, b, dim=-1)
print(similarity)