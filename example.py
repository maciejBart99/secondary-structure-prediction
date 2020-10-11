import torch
import torch.nn as nn

batch_size = 10
nb_classes = 2
nb_features = 5
nb_out = 10

x = torch.randn(batch_size, nb_features)
target = torch.randint(0, nb_classes, (batch_size,))

mapping = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = nn.Linear(nb_features, nb_out)
criterion = nn.CrossEntropyLoss()

output = model(x)
output_small = torch.zeros(batch_size, nb_classes).scatter_add(1, mapping.unsqueeze(0).expand_as(output), output)
loss = criterion(output_small, target)
loss.backward()
print(mapping.unsqueeze(0).unsqueeze(0))
print(output)
print(output_small)