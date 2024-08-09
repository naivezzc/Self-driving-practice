import torch

def image_gradients(image):
    # 在水平方向上计算梯度
    grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    # 在垂直方向上计算梯度
    grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]

    # 为了保持与原始图像相同的大小，填充最后一列和最后一行
    grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
    grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))

    return grad_x, grad_y


if __name__ == "__main__":
    input = torch.tensor([
        [0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.],
        [10., 11., 12., 13., 14.],
        [15., 16., 17., 18., 19.],
        [20., 21., 22., 23., 24.]
    ])

    input = input.unsqueeze(0).unsqueeze(0)
    dx, dy = image_gradients(input)
    print(input)
    print(dx)
    print(dy)
