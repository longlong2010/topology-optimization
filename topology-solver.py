import numpy;
import math;
import scipy.sparse;
import pyNastran.bdf.bdf;
import matplotlib.pyplot as plt;

def triaElement(nodes, property):
    Ke = numpy.zeros((6, 6), dtype=numpy.float32);
    [n1, n2, n3] = nodes;
    #节点坐标矩阵
    C = numpy.array([n1, n2, n3], dtype=numpy.float32);
    #高斯积分点
    points = [[1./3., 1./3., 0.5]];
    E = property[0];
    nu = property[1];
    t = property[2];
    #本构矩阵
    D = E / (1 - nu ** 2) * \
        numpy.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]], dtype=numpy.float32);
    for p in points:
        [xi, eta, w] = p;
        #单元坐标导数
        der = numpy.array([[1, 0, -1], [0, 1, -1]], dtype=numpy.float32);
        #雅可比矩阵
        J = der @ C;
        #全局坐标导数
        Der = numpy.linalg.inv(J) @ der;
        #应变矩阵
        B = numpy.zeros((3, 6), dtype=numpy.float32);
        for i in range(0, 3):
            B[0, i * 2] = Der[0, i];
            B[1, i * 2 + 1] = Der[1, i];
            B[2, i * 2] = Der[1, i];
            B[2, i * 2 + 1] = Der[0, i];
        Ke += B.T @ D @ B * abs(numpy.linalg.det(J)) * w * t;
    return Ke;


if __name__ == '__main__':
    # nodes = {1 : [300, 0], 2 : [300, 200], 3 : [0, 200], 4 : [0, 0]};
    # elements = [[1, 2, 4], [2, 3, 4]];
    # property = [206e3, 0.3, 10];

    property = [110e3, 0.3, 20];
    max_step = 20;
    #背景网格
    dx = dy = 1.5;
    x0 = y0 = 0;
    #节点及单元列表
    nodes = {};
    X = [];
    Y = [];
    TRIA = [];
    elements = [];
    #背景网格
    grids = {};
    #读取模型
    bdf = pyNastran.bdf.bdf.BDF(debug=False);
    bdf.read_bdf('Model1.bdf');
    for nid, n in bdf.nodes.items():
        nodes[nid] = n.xyz[0 : 2];
        X.append(n.xyz[0]);
        Y.append(n.xyz[1]);
    for eid, e in bdf.elements.items():
        elements.append(e.nodes);
        TRIA.append([e.nodes[0] - 1, e.nodes[1] - 1, e.nodes[2] - 1]);

    #单元中心添加到网格
    for idx, e in enumerate(elements):
        x = (nodes[e[0]][0] + nodes[e[1]][0] + nodes[e[2]][0]) / 3;
        y = (nodes[e[0]][1] + nodes[e[1]][1] + nodes[e[2]][1]) / 3;
        i = int((x - x0) // dx);
        j = int((y - y0) // dy);
        if (i, j) in grids:
            grids[(i, j)].append(idx);
        else:
            grids[(i, j)] = [idx];

    #单元总数
    N = len(elements);
    #自由度总数
    ndof = len(nodes) * 2;
    #优化参数
    volfrac = 0.5;
    Xe = volfrac * numpy.ones((N, ), dtype=numpy.float32);
    dC = numpy.zeros((N, ), dtype=numpy.float32);

    figure, ax = plt.subplots();
    for s in range(0, max_step):
        K = scipy.sparse.dok_matrix((ndof, ndof), dtype=numpy.float32);
        P = numpy.zeros((ndof, ), dtype=numpy.float32);
        #计算总体刚度矩阵
        for idx, e in enumerate(elements):
            Ke = triaElement([nodes[e[0]], nodes[e[1]], nodes[e[2]]], property) * (Xe[idx] ** 3);
            k = 0;
            #自由度映射表
            midof = dict();
            for n in e:
                n = n - 1;
                dofn = n * 2;
                midof[k] = dofn;
                k += 1;
                dofn += 1;
                midof[k] = dofn;
                k += 1;
            #单元刚度矩阵集成
            for i in range(0, 6):
                for j in range(0, 6):
                    mi = midof[i];
                    mj = midof[j];
                    K[mi, mj] += Ke[i, j];
        #施加位移边界条件
        for _, spcs in bdf.spcs.items():
            for spc in spcs:
                for n in spc.node_ids:
                    n = n - 1;
                    dofn = n * 2;
                    K[dofn, dofn] += 1e10;
                    K[dofn + 1, dofn + 1] += 1e10;

        # K[1, 1] += 1e20;
        # K[4, 4] += 1e20;
        # K[5, 5] += 1e20;
        # K[6, 6] += 1e20;
        # K[7, 7] += 1e20;

        #力边界条件
        for _, loads in bdf.loads.items():
            for load in loads:
                n = load.node_id - 1;
                dofn = n * 2;
                P[dofn] += load.xyz[0] * load.mag;
                P[dofn + 1] += load.xyz[1] * load.mag;

        # P[3] += -500e3;

        K = K.tocsr();
        U = scipy.sparse.linalg.spsolve(K, P);
        E0 = 0.;
        for jdx, e in enumerate(elements):
            Ke = triaElement([nodes[e[0]], nodes[e[1]], nodes[e[2]]], property);
            Ue = numpy.zeros((6, 1), dtype=numpy.float32);
            for idx, n in enumerate(e):
                n = n - 1;
                Ue[2 * idx] = U[2 * n];
                Ue[2 * idx + 1] = U[2 * n + 1];
            E = abs(0.5 * Ue.T @ Ke @ Ue);
            dC[jdx] = -3 * Xe[jdx] ** 2 * E;
            E0 += E;

        #密度平均化
        Rmin = 2;
        dCn = numpy.zeros_like(dC);
        for jdx, e in enumerate(elements):
            x = (nodes[e[0]][0] + nodes[e[1]][0] + nodes[e[2]][0]) / 3;
            y = (nodes[e[0]][1] + nodes[e[1]][1] + nodes[e[2]][1]) / 3;
            i = int((x - x0) // dx); #行指标
            j = int((y - y0) // dy); #列指标
            sum = 0.;
            for l in range(i - round(Rmin), i + round(Rmin)):
                for k in range(j - round(Rmin), j + round(Rmin)):
                    for idx in grids[(i, j)]:
                        e1 = elements[idx];
                        x1 = (nodes[e1[0]][0] + nodes[e1[1]][0] + nodes[e1[2]][0]) / 3;
                        y1 = (nodes[e1[0]][1] + nodes[e1[1]][1] + nodes[e1[2]][1]) / 3;
                        fac = max(0, Rmin * (dx ** 2 + dy ** 2) - math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2));
                        sum += fac;
                        dCn[jdx] += fac * Xe[idx] * dC[idx];
            dCn[jdx] /= (Xe[jdx] * sum);
        dC = dCn;

        #优化设计参数
        L1 = 0.;
        L2 = 1000000.;
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
        ax.set_aspect('equal');
        ax.tripcolor(X, Y, TRIA, facecolors=Xe);
        #plt.pause(0.1);
        plt.savefig(str(s).zfill(4) + '.png');
        Xe = nXe;
        print(E0, Xe.sum() / N);