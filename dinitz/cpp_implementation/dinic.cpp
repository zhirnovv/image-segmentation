#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cmath>
#include <experimental/filesystem>
#include <fstream>

using namespace std;

const long long INF = 1e18;
const int OBJECT_MARKER = 2;
const int BACKGROUND_MARKER = 1;
const long double NOISE_RATIO = 10;

struct Edge {
  int v, u;
  long double edgeCapacity, reverseFlow, flow = 0;
  Edge(int v, int u, long double cap) : v(v), u(u), edgeCapacity(cap) {
    reverseFlow = 0;
  }
};

struct GraphProcessor {
  vector<Edge> edges;
  vector<Edge> residualNetwork;
  vector<vector<int>> adjacencyList;
  int n, m = 0;
  int source, sink;
  vector<int> level, ptr;
  queue<int> bfsQueue;
  queue<int> minCutQueue;

  GraphProcessor() {}

  GraphProcessor(int n, int s, int t) : n(n), source(s), sink(t) {
    adjacencyList.resize(n);
    level.resize(n);
    ptr.resize(n);
  }

  void addEdge(int v, int u, long double cap) {
    residualNetwork.emplace_back(v, u, cap);
    residualNetwork.emplace_back(u, v, 0);
    edges.emplace_back(v, u, cap);
    edges.emplace_back(u, v, 0);
    adjacencyList[v].push_back(m);
    adjacencyList[u].push_back(m + 1);
    m += 2;
  }

  bool bfs() {
    while (!bfsQueue.empty()) {
      int v = bfsQueue.front();
      bfsQueue.pop();
      for (int id : adjacencyList[v]) {
        if (edges[id].edgeCapacity - edges[id].flow <= 0)
          continue;
        if (level[edges[id].u] != -1)
          continue;
        level[edges[id].u] = level[v] + 1;
        bfsQueue.push(edges[id].u);
      }
    }
    return level[sink] != -1;
  }

  long double dfs(int v, long double pushed) {
    /* cout << "DFS -> PUSHED: " << pushed << endl; */

    if (pushed == 0)
      return 0;

    if (v == sink)
      return pushed;

    for (int &cid = ptr[v]; cid < (int)adjacencyList[v].size(); cid++) {
      int id = adjacencyList[v][cid];
      int u = edges[id].u;
      /* cout << "EDGE capacity and flow: " << edges[id].edgeCapacity << ' ' << edges[id].flow << endl; */
      if (level[v] + 1 != level[u] ||
          edges[id].edgeCapacity - edges[id].flow <= 0)
        continue;
      long double pushedFlow =
          dfs(u, min(pushed, edges[id].edgeCapacity - edges[id].flow));
      if (pushedFlow == 0)
        continue;
      edges[id].flow += pushedFlow;
      edges[id ^ 1].flow -= pushedFlow;
      return pushedFlow;
    }

    return 0;
  }

  long double flow() {
    long double maxFlow = 0;

    while (true) {
      fill(level.begin(), level.end(), -1);
      level[source] = 0;
      bfsQueue.push(source);
      if (!bfs())
        break;
      fill(ptr.begin(), ptr.end(), 0);

      long double pushed = dfs(source, INF);
      /* cout << "PUSHED: " << pushed << ' ' << (pushed == 0 ? "true" : "false") << endl << endl; */

      while (pushed) {
        /* cout << "MAX FLOW: " << maxFlow << " + " << pushed << endl; */
        maxFlow += pushed;
        pushed = dfs(source, INF);
      }
    }

    return maxFlow;
  }

  void buildResidualNetwork() {
    for (int i = 0; i < residualNetwork.size(); i++) {
      if (edges[i].flow >= 0) {
        residualNetwork[i].flow =
            residualNetwork[i].edgeCapacity + edges[i ^ 1].flow;
      } else {
        residualNetwork[i].flow = abs(edges[i].flow);
      }
    }
  }

  vector<long double> minCutBfs(vector<Edge> viableEdges) {
    vector<bool> isVisited;
    isVisited.resize(n, false);

    isVisited[source] = true;

    vector<long double> minCut;

    while (!minCutQueue.empty()) {
      int v = minCutQueue.front();
      minCutQueue.pop();

      minCut.push_back(v);

      vector<Edge> adjacentEdges;

      copy_if(viableEdges.begin(), viableEdges.end(),
              back_inserter(adjacentEdges), [v](Edge potentialEdge) {
                return potentialEdge.flow > 0 && potentialEdge.v == v;
              });

      for (int i = 0; i < adjacentEdges.size(); i++) {
        int to = adjacentEdges[i].u;
        long double flow = adjacentEdges[i].flow;
        if (flow > 0 && !isVisited[to]) {
          isVisited[to] = true;
          minCutQueue.push(to);
        }
      }
    }

    isVisited.clear();
    isVisited.shrink_to_fit();

    return minCut;
  }

  vector<long double> minCut() {
    buildResidualNetwork();

    vector<Edge> viableEdges;

    copy_if(residualNetwork.begin(), residualNetwork.end(),
            back_inserter(viableEdges),
            [](Edge potentialEdge) { return potentialEdge.flow > 0; });

    minCutQueue.push(source);

    return minCutBfs(viableEdges);
  }
};

struct ImageMask {
  vector<vector<int>> maskMatrix;
  int rows, cols;

  ImageMask(int rows, int cols) : rows(rows), cols(cols) {
    maskMatrix.resize(rows, vector<int>(cols));
  };

  void addObjectMarker(int x, int y) { maskMatrix[x][y] = OBJECT_MARKER; }

  void addBackgroundMarker(int x, int y) {
    maskMatrix[x][y] = BACKGROUND_MARKER;
  }
};

struct ImageProcessor {
  string imagePath;
  cv::Mat image;
  GraphProcessor graph;
  ImageMask imageMask;

  long double getPixelEdgeCapacity(cv::Point start, cv::Point end) {
    double distance = cv::norm(cv::Mat(start), cv::Mat(end));
    cv::Scalar startIntensity = image.at<uchar>(start);
    cv::Scalar endIntensity = image.at<uchar>(end);
    /* cout << "coords: " << start.x << start.y << ' ' << end.x << end.y << endl; */
    /* cout << "intensity: " << startIntensity.val[0] << ' ' << endIntensity.val[0] << ' ' << pow(startIntensity.val[0] - endIntensity.val[0], 2) */
    /*      << endl; */
    /* cout << "distance: " << distance << endl; */
    /* cout << exp(-1 * pow(startIntensity.val[0] - endIntensity.val[0], 2) / */
    /*             (2 * pow(NOISE_RATIO, 2))) / */
    /*             distance << endl; */

    return exp(-1 * pow(startIntensity.val[0] - endIntensity.val[0], 2) /
               (2 * pow(NOISE_RATIO, 2))) /
           distance;
  }

  int getPixelIndex(int x, int y) { return (y) * image.cols + x; }

  ImageProcessor(string imagePath, ImageMask mask)
      : imagePath(imagePath), imageMask(mask) {
    image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    /* cout << "BRUH 1" << endl; */

    if (image.empty()) {
      throw runtime_error(
          "Could not load image. Please check if the image path is correct.");
    }

    int imageSize = image.rows * image.cols + 2;
    graph = GraphProcessor(imageSize, imageSize - 2, imageSize - 1);

    /* cv::imshow("Seed input", image); */
    /* int key = cv::waitKey(0); */

    /* cout << "BRUH 2" << endl; */

    for (int i = 0; i < imageMask.rows; i++) {
      for (int j = 0; j < imageMask.cols; j++) {
        int maskIndex = getPixelIndex(i, j);
        switch (imageMask.maskMatrix[i][j]) {
        case (OBJECT_MARKER):
          graph.addEdge(imageSize - 2, maskIndex, INF);
          break;
        case (BACKGROUND_MARKER):
          graph.addEdge(maskIndex, imageSize - 1, INF);
        default:
          break;
        }
      }
    }

    /* cout << "BRUH 3" << endl; */

    cv::MatIterator_<uchar> it, end;
    cv::Point pixelPosition, rightPixelPosition, leftPixelPosition,
        topPixelPosition, bottomPixelPosition;
    int pixelIndex, rightPixelIndex, leftPixelIndex, topPixelIndex,
        bottomPixelIndex;

    /* cout << "BRUH 4" << endl; */

    for (it = image.begin<uchar>(), end = image.end<uchar>(); it < end; ++it) {
      pixelPosition = it.pos();
      /* cout << pixelPosition.x << ' ' << pixelPosition.y << endl; */
      rightPixelPosition = cv::Point(pixelPosition.x + 1, pixelPosition.y);
      leftPixelPosition = cv::Point(pixelPosition.x - 1, pixelPosition.y);
      topPixelPosition = cv::Point(pixelPosition.x, pixelPosition.y - 1);
      bottomPixelPosition = cv::Point(pixelPosition.x, pixelPosition.y + 1);

      pixelIndex = getPixelIndex(pixelPosition.x, pixelPosition.y);
      rightPixelIndex =
          getPixelIndex(rightPixelPosition.x, rightPixelPosition.y);
      leftPixelIndex = getPixelIndex(leftPixelPosition.x, leftPixelPosition.y);
      topPixelIndex = getPixelIndex(topPixelPosition.x, topPixelPosition.y);
      bottomPixelIndex =
          getPixelIndex(bottomPixelPosition.x, bottomPixelPosition.y);

      if (topPixelPosition.y >= 0) {
        /* cout << "top" << endl; */
        graph.addEdge(pixelIndex, topPixelIndex,
                      getPixelEdgeCapacity(pixelPosition, topPixelPosition));
      }

      if (bottomPixelPosition.y <= image.rows - 1) {
        /* cout << "bottom" << endl; */
        /* cout << pixelIndex << ' ' << bottomPixelIndex << endl; */
        graph.addEdge(pixelIndex, bottomPixelIndex,
                      getPixelEdgeCapacity(pixelPosition, bottomPixelPosition));
      }

      if (leftPixelPosition.x >= 0) {
        /* cout << "left" << endl; */
        graph.addEdge(pixelIndex, leftPixelIndex,
                      getPixelEdgeCapacity(pixelPosition, leftPixelPosition));
      }

      if (rightPixelPosition.x <= image.cols - 1) {
        /* cout << "right" << endl; */
        graph.addEdge(pixelIndex, rightPixelIndex,
                      getPixelEdgeCapacity(pixelPosition, rightPixelPosition));
      }
    }

    /* for (int i = 0; i < graph.edges.size(); i++) { */
    /*   cout << graph.edges[i].v << ' ' << graph.edges[i].u << ' ' << graph.edges[i].edgeCapacity << endl; */
    /* } */

    cout << graph.flow() << endl;

    vector<long double> cut = graph.minCut();

    for (int i = 0; i < cut.size(); i++) {
      cout << cut[i] << ' ';
    }
  }
};

int main() {
  ImageMask mask = ImageMask(7, 7);
  mask.addBackgroundMarker(0, 0);
  mask.addObjectMarker(3, 4);
  /* mask.addObjectMarker(21, 22); */
  /* mask.addObjectMarker(22, 13); */

  ImageProcessor processor("./tests/images/test_2.jpg", mask);

  return 0;
  /* string path = "./tests/dinic"; */

  /* int vertices, edges; */
  /* int start, end; */
  /* long double capacity; */
  /* chrono::time_point<chrono::system_clock> startTime, endTime; */

  /* for (const auto &entry : filesystem::directory_iterator(path)) { */
  /*   if (entry.path() != "./tests/dinic/test_kek2.txt") { */
  /*     continue; */
  /*   } */

  /*   cout << entry.path() << endl; */

  /*   ifstream infile(entry.path()); */

  /*   string line; */
  /*   getline(infile, line); */

  /*   istringstream graphParams(line); */
  /*   if (!(graphParams >> vertices >> edges)) { */
  /*     break; */
  /*   } */

  /*   GraphProcessor graph(vertices, 0, vertices - 1); */

  /*   for (int i = 0; i < edges; i++) { */
  /*     getline(infile, line); */

  /*     istringstream edgeParams(line); */
  /*     if (!(edgeParams >> start >> end >> capacity)) { */
  /*       break; */
  /*     } */
  /*     graph.addEdge(start - 1, end - 1, capacity); */
  /*   } */

  /*   startTime = chrono::system_clock::now(); */

  /*   cout << "Max flow is: " << graph.flow() << endl; */

  /*   vector<int> cut = graph.minCut(); */

  /*   for (int i = 0; i < cut.size(); i++) { */
  /*     cout << cut[i] + 1 << ' '; */
  /*   } */

  /*   cout << endl; */

  /*   endTime = chrono::system_clock::now(); */

  /*   cout << "Elapsed time: " */
  /*        << chrono::duration_cast<chrono::microseconds>(endTime - startTime)
   */
  /*               .count() */
  /*        << endl */
  /*        << endl; */
  /* } */

  /* return 0; */
}
