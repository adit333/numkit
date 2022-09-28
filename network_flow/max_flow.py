""" Includes implementation of Edmonds-Karp and Dinic's algorithm for finding
    maximum flow in a flow network.

References: Introduction to Algorithms, Cormen et al.
The variable names have purposely been kept similar to the ones used in the
book. This code is optimized for readability along with the book.
"""


from numbers import Number
from copy import deepcopy

INF = float('inf')


class MaximumFlow:
    def __init__(self, vertices: int):
        """
        Creates a new instance of the MaxFlow class. The general way you'll
        want to use this library is to create a new instance of the class,
        add edges, then call the `edmonds_karp` or `dinic` methods.
        While the library does support floats, be aware that it is not advised
        to use due to the potential for floating point errors, meaning small
        amounts of flow may be sent many times.

        Arguments:
            V: The number of vertices in the graph.

        Example:
            >>> mf = MaxFlow(3)
            >>> mf.add_edge(0, 1, 3)
            >>> mf.add_edge(0, 2, 3)
            >>> mf.add_edge(1, 2, 3)
            >>> mf.dinic(0, 2)  # or mf.edmonds_karp(0, 2)
            6
        """
        self.vertices = vertices
        self.edge_list = []
        self.adjacency_list = [list() for _ in range(self.vertices)]
        self.distances = []
        self.last = []
        self.points = []
        self.has_been_run = False

    def BFS(self, s: int, t: int) -> bool:
        self.distances = [-1] * self.vertices
        self.distances[s] = 0
        self.points = [[-1, -1] for _ in range(self.vertices)]
        q = [s]
        while len(q) != 0:
            u = q[0]
            q.pop(0)
            if u == t:
                break
            for idx in self.adjacency_list[u]:
                v, cap, flow = self.edge_list[idx]
                if cap - flow > 0 and self.distances[v] == -1:
                    self.distances[v] = self.distances[u]+1
                    q.append(v)
                    self.points[v] = [u, idx]
        return self.distances[t] != -1

    def send_one_flow(self, s: int, t: int, f: Number = INF) -> Number:
        if s == t:
            return f
        u, idx = self.points[t]
        _, cap, flow = self.edge_list[idx]
        pushed = self.send_one_flow(s, u, min(f, cap-flow))
        flow += pushed
        self.edge_list[idx][2] = flow
        self.edge_list[idx ^ 1][2] -= pushed
        return pushed

    def DFS(self, u: int, t: int, f: Number = INF) -> Number:
        if u == t or f == 0:
            return f
        for i in range(self.last[u], len(self.adjacency_list[u])):
            self.last[u] = i
            v, cap, flow = self.edge_list[self.adjacency_list[u][i]]
            if self.distances[v] != self.distances[u]+1:
                continue
            pushed = self.DFS(v, t, min(f, cap - flow))
            if pushed != 0:
                flow += pushed
                self.edge_list[self.adjacency_list[u][i]][2] = flow
                self.edge_list[self.adjacency_list[u][i] ^ 1][2] -= pushed
                return pushed
        return 0

    def add_edge(self, u: int, v: int, capacity: Number,
                 directed: bool = True) -> None:
        """
        Adds an edge from `u` to `v` with capacity `w`. By default, the edge is
        directed, i.e. `u`->`v`. You can set `directed = False` to add it
        as an undirected edge `u`<->`v`.

        Arguments:
            `u`: The first vertex.
            `v`: The second vertex.
            `capacity`: The capacity of the edge.
            `directed`: Whether the edge is directed. True by default.

        Example:
            >>> mf = MaxFlow(3)
            >>> mf.add_edge(0, 1, 3)
            >>> mf.add_edge(2, 1, 3)
        """
        if u == v:
            return
        self.edge_list.append([v, capacity, 0])
        self.adjacency_list[u].append(len(self.edge_list)-1)
        self.edge_list.append([u, 0 if directed else capacity, 0])
        self.adjacency_list[v].append(len(self.edge_list)-1)

    def assert_has_not_already_been_run(self):
        if self.has_been_run:
            msg = ('Rerunning a max flow algorithm on the same graph will '
                   + 'result in incorrect behaviour. Please use .copy() '
                   + 'before you run any max flow algorithm if you need to '
                   + 'run multiple iterations')
            raise Exception(msg)

        self.has_been_run = True

    def edmonds_karp(self, s: int, t: int) -> Number:
        """
        Returns the max flow obtained by running Edmons-Karp algorithm.
        Modifies the graph in place.

        Arguments:
            `s`: The source vertex.
            `t`: The sink vertex.

        Returns:
            The max flow.
        """
        self.assert_has_not_already_been_run()

        mf = 0
        while self.BFS(s, t):
            f = self.send_one_flow(s, t)
            if f == 0:
                break
            mf += f
        return mf

    def dinic(self, s: int, t: int) -> Number:
        """
        Returns the max flow obtained by running Dinic's algorithm.
        Modifies the graph in place.

        Arguments:
            `s`: The source vertex.
            `t`: The sink vertex.

        Returns:
            The max flow.
        """
        self.assert_has_not_already_been_run()

        mf = 0
        while self.BFS(s, t):
            self.last = [0] * self.vertices
            f = self.DFS(s, t)
            while f != 0:
                mf += f
                f = self.DFS(s, t)
        return mf

    def copy(self) -> 'MaximumFlow':
        """
        Returns a deep copy of the current instance. This is convenient for
        problems where you need to run MaxFlow multiple times on slightly
        different graphs, since the instance is destroyed after each max flow
        run.

        Example:
            >>> mf = MaxFlow(4)
            >>> mf.add_edge(0, 1, 3)
            >>> mf.add_edge(1, 2, 3)
            >>> for c in range(1, 4):
            >>>     mf_copy = mf.copy()
            >>>     mf_copy.add_edge(2, 3, c)
            >>>     res = mf_copy.dinic(0, 3)  # Will not modify mf
        """
        return deepcopy(self)

    def __repr__(self) -> str:
        el = self.edge_list[:10] + ['...'] if len(self.edge_list) > 10 else self.edge_list
        al = self.adjacency_list[:10] + ['...'] if len(self.adjacency_list) > 10 else self.adjacency_list
        el = ', '.join(map(str, el))
        al = ', '.join(map(str, al))
        return f'MaxFlow(V={self.vertices}, EL=[{el}], AL=[{al}])'


def main():
    fp = open('maxflow_in.txt', 'r')
    V, s, t = map(int, fp.readline().strip().split())
    mf = MaximumFlow(V)

    for u in range(V):
        tkn = list(map(int, fp.readline().strip().split()))
        k = tkn[0]
        for i in range(k):
            v, w = tkn[2*i+1], tkn[2*i+2]
            mf.add_edge(u, v, w)

    # print(mf.edmonds_karp(s, t))
    print(mf.dinic(s, t))


if __name__ == '__main__':
    main()
