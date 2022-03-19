import numpy;
import math;
import scipy.sparse;
import scipy.sparse.linalg;
import matplotlib.pyplot as plt;

def quadElement(nodes, property):
    Ke = numpy.zeros((8, 8), dtype=numpy.float32);
    [n1, n2, n3, n4] = nodes;
    #节点坐标矩阵
    C = numpy.array([n1, n2, n3, n4], dtype=numpy.float32);
    #高斯积分点
    points = [[1 / math.sqrt(3), 1 / math.sqrt(3), 1], 
              [-1 / math.sqrt(3), 1 / math.sqrt(3), 1], 
              [1 / math.sqrt(3), -1 / math.sqrt(3), 1], 
              [-1 / math.sqrt(3), -1 / math.sqrt(3), 1]];
    E = property[0];
    nu = property[1];
    t = property[2];
    #本构矩阵
    D = E / (1 - nu ** 2) * \
        numpy.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]], dtype=numpy.float32);
    for p in points:
        [xi, eta, w] = p;
        #单元坐标导数
        der = numpy.array([[-0.25 * (1 - eta), -0.25 * (1 + eta), 0.25 * (1 + eta), 0.25 * (1 - eta)], 
                          [-0.25 * (1 - xi), 0.25 * (1 - xi), 0.25 * (1 + xi), -0.25 * (1 + xi)]], dtype=numpy.float32);
        #雅可比矩阵
        J = der @ C;
        #全局坐标导数
        Der = numpy.linalg.inv(J) @ der;
        #应变矩阵
        B = numpy.zeros((3, 8), dtype=numpy.float32);
        for i in range(0, 4):
            B[0, i * 2] = Der[0, i];
            B[1, i * 2 + 1] = Der[1, i];
            B[2, i * 2] = Der[1, i];
            B[2, i * 2 + 1] = Der[0, i];
        Ke += B.T @ D @ B * abs(numpy.linalg.det(J)) * w * t;
    return Ke;

if __name__ == '__main__':
    #材料及单元属性
    property = [1, 0.3, 5];
    #网格尺度
    dx = 1;
    dy = 1;
    w = 121;
    h = 61;
    #节点及单元列表
    nodes = [];
    elements = [];
    #划分节点
    for j in range(0, h):
        for i in range(0, w):
            x = i * dx;
            y = -j * dy;
            node = [x, y];
            nodes.append(node);
    #划分单元
    for j in range(0, h - 1):
        for i in range(0, w - 1):
            n1 = i + (j + 1) * w;
            n2 = i + j * w;
            n3 = (i + 1) + j * w;
            n4 = (i + 1) + (j + 1) * w;
            element = [n1, n2, n3, n4];
            elements.append(element);
    #单元总数
    N = len(elements);
    #计算总体刚度矩阵
    ndof = len(nodes) * 2;
    volfrac = 0.6;
    Xe = volfrac * numpy.ones((N, 1), dtype=numpy.float32);
    dC = numpy.zeros((N, 1), dtype=numpy.float32);
    figure, ax = plt.subplots();
    for s in range(0, 1000):
        K = scipy.sparse.dok_matrix((ndof, ndof), dtype=numpy.float32)
        P = numpy.zeros((ndof, 1), dtype=numpy.float32);
        for idx, e in enumerate(elements):
            Ke = quadElement([nodes[e[0]], nodes[e[1]], nodes[e[2]], nodes[e[3]]], property) * (Xe[idx] ** 3);
            k = 0;
            #自由度映射表
            midof = dict();
            for n in e:
                dofn = n * 2;
                midof[k] = dofn;
                k += 1;
                dofn += 1;
                midof[k] = dofn;
                k += 1;
            #单元刚度矩阵集成
            for i in range(0, 8):
                for j in range(0, 8):
                    mi = midof[i];
                    mj = midof[j];
                    K[mi, mj] += Ke[i, j];
        #施加位移边界条件
        for j in range(0, h):
            n = j * w;
            dofn = n * 2;
            K[dofn, dofn] += 1e10;
            K[dofn + 1, dofn + 1] += 1e10;
        #力边界条件
        #P[2 * (w - 1) + 1, 0] = 1;
        for j in range(0, h):
           P[(j * w + w - 1) * 2 + 1] = 1 / h;
           #P[(j * w + w - 1) * 2] = 1 / h;
        #求解位移
        K = K.tocsr();
        U = scipy.sparse.linalg.spsolve(K, P);
        E0 = 0.
        for jdx, e in enumerate(elements):
            Ke = quadElement([nodes[e[0]], nodes[e[1]], nodes[e[2]], nodes[e[3]]], property);
            Ue = numpy.zeros((8, 1), dtype=numpy.float32);
            for idx, n in enumerate(e):
                Ue[2 * idx] = U[2 * n];
                Ue[2 * idx + 1] = U[2 * n + 1];
            E = abs(0.5 * Ue.T @ Ke @ Ue);
            dC[jdx] = -3 * Xe[jdx] ** 2 * E;
            E0 += E;

        #密度平均化
        Rmin = 1.5;
        dCn = numpy.zeros_like(dC);
        for jdx in range(0, N):
            i = jdx // (w - 1); #行指标
            j = jdx % (w - 1);  #列指标
            sum = 0.;
            for l in range(max(0, i - round(Rmin)), min(i + round(Rmin), h - 1)):
                for k in range(max(0, j - round(Rmin)), min(j + round(Rmin), w - 1)):
                    fac = max(0, Rmin - math.sqrt((j - k) ** 2 + (l - i) ** 2));
                    sum += fac
                    idx = l * (w - 1) + k;
                    dCn[jdx] += fac * Xe[idx] * dC[idx];
            dCn[jdx] /= (Xe[jdx] * sum);
        dC = dCn;

        #优化设计参数
        L1 = 0;
        L2 = 1000000;
        M = 0.2;
        eps = 1e-4
        Xmin = 1e-3;
        nXe = numpy.zeros_like(Xe);
        while L2 - L1 > eps:
            Lm = (L1 + L2) / 2;
            for i in range(0, N):
                nXe[i] = max(Xmin, max(Xe[i] - M, min(1., min(Xe[i] + M, Xe[i] * numpy.sqrt(abs(dC[i] / Lm))))));
            #二分法搜索乘子
            if nXe.sum() > volfrac * N:
                L1 = Lm;
            else:
                L2 = Lm;

        #结果可视化
        harvest = numpy.zeros((h - 1, w - 1), numpy.float32);
        for j in range(0, h - 1):
            for i in range(0, w - 1):
                k = j * (w - 1) + i;
                harvest[j, i] = Xe[k];
        im = ax.imshow(harvest);
        #plt.savefig(str(s).zfill(4) + '.png');
        plt.pause(0.1);
        #输出变形能及密度比
        Xe = nXe;
        print(E0, Xe.sum() / N);
