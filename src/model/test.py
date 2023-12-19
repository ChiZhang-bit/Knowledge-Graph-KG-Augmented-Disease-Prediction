import torch
# a = torch.randn(3,3,3,3)

# for i in range(a.shape[2]):
#     print(a[:, :, i, :].shape)

ua = torch.nn.Parameter(torch.randn(5))
a = torch.randn(3,4,5)
print(a.shape)
print(ua.shape)
b = torch.matmul(a, ua)
print(b.unsqueeze(2).shape)



attention_weights = torch.zeros(8,6,256)  # (batch_size, visit_num, 256)
for i in range(256):
    b = torch.randn(8,6)
    print(b.shape)
    print(attention_weights[:, :, i].shape)
    attention_weights[:, :, i] = b  # (batch_size, visit_num)