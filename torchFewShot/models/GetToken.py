import torch
import torch.nn as nn

class Token(nn.Module):
    def __init__(self, x=13):
        super(Token, self).__init__()
        self.token0 = nn.Parameter(torch.randn(x))
        self.token1 = nn.Parameter(torch.randn(x))
        self.token2 = nn.Parameter(torch.randn(x))
        self.token3 = nn.Parameter(torch.randn(x))
        self.token4 = nn.Parameter(torch.randn(x))

    def forward(self, picture):
        support_token = torch.zeros((375, 5, 64),requires_grad=False).cuda()
        idx_match = [range(0, 13), range(12, 25), range(25, 38), range(38, 51), range(51, 64)]
        token = [self.token0, self.token1, self.token2, self.token3, self.token4]
        for i in range(375):
            # get class_token_match
            distance = torch.zeros((5, 5, 13)).cuda()
            distance_absolute0 = torch.zeros((5, 5)).cuda()
            for j in range(5):
                for k in range(5):
                    distance[j, k, :] = picture[i, j, idx_match[k]] - token[k]
                    distance_absolute0[j, k] = torch.sum(torch.pow(picture[i, j, :], 2)) - torch.sum(
                        torch.pow(picture[i, j, idx_match[k]], 2)) + torch.sum(torch.pow(distance[j, k, :], 2))
            idx0 = torch.argmin(distance_absolute0).item()
            class0, token0 = torch.div(idx0, 5, rounding_mode='trunc').item(), (idx0 - torch.div(idx0, 5, rounding_mode='trunc') * 5).item()
            find_min = torch.tensor([1000000]).cuda()
            idx_x = [0, 1, 2, 3, 4]
            idx_y = [0, 1, 2, 3, 4]
            idx_x.remove(class0)
            idx_y.remove(token0)
            for l in idx_x:
                for m in idx_y:
                    if distance_absolute0[l, m] < find_min:
                        find_min = distance_absolute0[l, m]
            for n in idx_x:
                for o in idx_y:
                    if distance_absolute0[n, o] == find_min:
                        class1, token1 = n, o
            find_min = torch.tensor([1000000]).cuda()
            idx_x.remove(class1)
            idx_y.remove(token1)
            for l in idx_x:
                for m in idx_y:
                    if distance_absolute0[l, m] < find_min:
                        find_min = distance_absolute0[l, m]
            for n in idx_x:
                for o in idx_y:
                    if distance_absolute0[n, o] == find_min:
                        class2, token2 = n, o
            find_min = torch.tensor([1000000]).cuda()
            idx_x.remove(class2)
            idx_y.remove(token2)
            for l in idx_x:
                for m in idx_y:
                    if distance_absolute0[l, m] < find_min:
                        find_min = distance_absolute0[l, m]
            for n in idx_x:
                for o in idx_y:
                    if distance_absolute0[n, o] == find_min:
                        class3, token3 = n, o
            find_min = torch.tensor([1000000]).cuda()
            idx_x.remove(class3)
            idx_y.remove(token3)
            class4, token4 = idx_x[0], idx_y[0]
            # get token_sample

            support_token[i, class0, idx_match[class0]] += token[token0]
            support_token[i, class1, idx_match[class1]] += token[token1]
            support_token[i, class2, idx_match[class2]] += token[token2]
            support_token[i, class3, idx_match[class3]] += token[token3]
            support_token[i, class4, idx_match[class4]] += token[token4]
        return support_token

if __name__=='__main__':
    a = Token().cuda()
    b = torch.randn(375,5,64).cuda()
    c=  a(b)
    print(c.shape)
    c = nn.Linear