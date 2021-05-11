#include<bits/stdc++.h>
#include<fstream>
#include<filesystem>
#include<chrono>

using namespace std;
 
struct FlowEdge {
    int v, u;
    long long cap, flow_reverse, flow = 0;
    FlowEdge(int v, int u, long long cap) : v(v), u(u), cap(cap) {
        flow_reverse = 0;
    }
};

struct Dinic {
    const long long flow_inf = 1e18;
    vector<FlowEdge> edges;
    vector<vector<int>> adj;
    int n, m = 0;
    int s, t;
    vector<int> level, ptr;
    queue<int> q;

    Dinic(int n, int s, int t) : n(n), s(s), t(t) {
        adj.resize(n);
        level.resize(n);
        ptr.resize(n);
    }

    void add_edge(int v, int u, long long cap) {
        edges.emplace_back(v, u, cap);
        edges.emplace_back(u, v, 0);
        adj[v].push_back(m);
        adj[u].push_back(m + 1);
        m += 2;
    }

    bool bfs() {
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (int id : adj[v]) {
                if (edges[id].cap - edges[id].flow < 1)
                    continue;
                if (level[edges[id].u] != -1)
                    continue;
                level[edges[id].u] = level[v] + 1;
                q.push(edges[id].u);
            }
        }
        return level[t] != -1;
    }

    long long dfs(int v, long long pushed) {
        if (pushed == 0)
            return 0;
        if (v == t)
            return pushed;
        for (int& cid = ptr[v]; cid < (int)adj[v].size(); cid++) {
            int id = adj[v][cid];
            int u = edges[id].u;
            if (level[v] + 1 != level[u] || edges[id].cap - edges[id].flow < 1)
                continue;
            long long tr = dfs(u, min(pushed, edges[id].cap - edges[id].flow));
            if (tr == 0)
                continue;
            edges[id].flow += tr;
            edges[id ^ 1].flow -= tr;
            return tr;
        }
        return 0;
    }

    long long flow() {
        long long f = 0;
        while (true) {
            fill(level.begin(), level.end(), -1);
            level[s] = 0;
            q.push(s);
            if (!bfs())
                break;
            fill(ptr.begin(), ptr.end(), 0);
            while (long long pushed = dfs(s, flow_inf)) {
                f += pushed;
            }
        }
        return f;
    }
};

int main()
{
    string path = "./tests/dinic";

    int vertices, edges;
    int start, end, capacity;
    chrono::time_point<chrono::system_clock> startTime, endTime; 

    for (const auto & entry : filesystem::directory_iterator(path)) {
        cout << entry.path() << endl;
        ifstream infile(entry.path());

        string line;
        getline(infile, line);


        istringstream graphParams(line);
        if (!(graphParams >> vertices >> edges)) {
            break;
        }

        Dinic graph(vertices, 0, vertices - 1);

        for (int i = 0; i < edges; i++) {
            getline(infile, line);

            istringstream edgeParams(line);
            if (!(edgeParams >> start >> end >> capacity)) {
                break;
            }
            graph.add_edge(start - 1, end - 1, capacity);
        }

        startTime = chrono::system_clock::now();

        cout << "Max flow is: " << graph.flow() << endl;

        endTime = chrono::system_clock::now();

        cout << "Elapsed time: " << chrono::duration_cast<chrono::microseconds>(endTime - startTime).count() << endl << endl;
    }

    return 0;
}
