import torch

input = torch.tensor([0.0, -4.0, -12.0], requires_grad=True)

grad_data = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

y = torch.abs(input)

y.backward(gradient=grad_data)

gradient = input.grad


print("Original tensor:", input)
print("Tensor after abs:", y)
print("Gradient of the original tensor:", gradient)

result = torch.where(gradient > 0 , torch.tensor(1.0), torch.where(gradient < 0, torch.tensor(-1.0), torch.tensor(0.0)))
print("calculated gradient:", result)

if (torch.all(gradient.eq(result))):
    print("Both tensors matched")