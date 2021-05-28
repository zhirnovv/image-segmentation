#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cmath>
#include <experimental/filesystem>
#include <fstream>

using namespace std;

const long long INF = 1e18;

const int OBJECT_MARKER = 2;
const int BACKGROUND_MARKER = 1;
const long double LAMBDA = 60.0;

const long double NOISE_RATIO = 0.7;

struct Edge {
  int v, u;
  long double edgeCapacity, flow = 0.0;
  Edge(int v, int u, long double cap) : v(v), u(u), edgeCapacity(cap) {}
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

  void adjustEdge(int v, int u, long double newCapacity) {
    for (int id : adjacencyList[v]) {
      if (edges[id].u == u && edges[id].edgeCapacity != 0) {
        edges[id].edgeCapacity = newCapacity;
        residualNetwork[id].edgeCapacity = newCapacity;
        return;
      }
    }
  }

  Edge getEdge(int v, int u) {
    for (int id : adjacencyList[v]) {
      if (edges[id].u == u) {
        return edges[id];
      }
    }

    return Edge(-1, -1, -1);
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
    if (pushed == 0)
      return 0;

    if (v == sink)
      return pushed;

    for (int &cid = ptr[v]; cid < (int)adjacencyList[v].size(); cid++) {
      int id = adjacencyList[v][cid];
      int u = edges[id].u;
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

  long double flow(long double startingFlow = INF) {
    for (int i; i < edges.size(); i++) {
      edges[i].flow = 0.0;
    }

    long double maxFlow = 0.0;

    while (true) {
      fill(level.begin(), level.end(), -1);
      level[source] = 0;
      bfsQueue.push(source);
      if (!bfs())
        break;
      fill(ptr.begin(), ptr.end(), 0);

      long double pushed = dfs(source, startingFlow);

      while (pushed) {
        maxFlow += pushed;
        pushed = dfs(source, startingFlow);
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

      int to;
      long double flow;

      for (int id : adjacencyList[v]) {
        to = residualNetwork[id].u;
        flow = residualNetwork[id].flow;

        if (flow <= 0) {
          continue;
        }

        if (isVisited[to]) {
          continue;
        }

        isVisited[to] = true;
        minCutQueue.push(to);
      }
    }

    cout << "Min cut size" << minCut.size() << endl;

    return minCut;
  }

  vector<long double> minCut() {
    for (int i = 0; i < residualNetwork.size(); i++) {
      residualNetwork[i].flow = 0.0;
    }

    buildResidualNetwork();

    vector<Edge> viableEdges;

    copy_if(residualNetwork.begin(), residualNetwork.end(),
            back_inserter(viableEdges),
            [](Edge potentialEdge) { return potentialEdge.flow > 0; });

    minCutQueue = queue<int>();

    minCutQueue.push(source);

    return minCutBfs(viableEdges);
  }
};

struct ImageMask {
  cv::Mat image;
  vector<vector<int>> maskMatrix;
  vector<int> objectHistogram;
  vector<int> backgroundHistogram;
  int objectMarkerCount = 0;
  int backgroundMarkerCount = 0;

  int currentEditorMarker = OBJECT_MARKER;
  bool isDrawing = false;

  ImageMask() {}

  ImageMask(cv::Mat image) : image(image) {
    maskMatrix.resize(image.cols, vector<int>(image.rows));
    objectHistogram.resize(256, 0);
    backgroundHistogram.resize(256, 0);

    buildMask();
  };

  void placeMarker(int x, int y, int marker) {
    if (x > image.cols - 1 || y > image.rows - 1 || x < 0 || y < 0) {
      return;
    }

    maskMatrix[x][y] = marker;

    cv::Scalar intensity = image.at<uchar>(y, x);

    if (marker == OBJECT_MARKER) {
      objectHistogram[intensity.val[0]]++;
      objectMarkerCount++;
    } else {
      backgroundHistogram[intensity.val[0]]++;
      backgroundMarkerCount++;
    }
  }

  static void handleEditorMouseEvents(int event, int x, int y, int flags,
                                      void *userdata) {
    ImageMask *imageMask = static_cast<ImageMask *>(userdata);

    switch (event) {
    case cv::EVENT_MOUSEMOVE:
      if (imageMask->isDrawing) {
        imageMask->placeMarker(x, y, imageMask->currentEditorMarker);
      }
      break;
    case cv::EVENT_LBUTTONDOWN:
      imageMask->isDrawing = true;
      imageMask->placeMarker(x, y, imageMask->currentEditorMarker);
      break;
    case cv::EVENT_LBUTTONUP:
      imageMask->isDrawing = false;
      break;
    default:
      break;
      cv::imshow("Marker Selector", imageMask->image);
    }
  }

  void buildMask() {
    cv::namedWindow("Marker Selector", cv::WINDOW_AUTOSIZE);
    cv::imshow("Marker Selector", image);
    cv::setMouseCallback("Marker Selector", handleEditorMouseEvents, this);

    while (true) {
      int key = cv::waitKey();

      switch (key) {
      case 111:
        currentEditorMarker = OBJECT_MARKER;
        cout << "Using object marker" << endl;
        break;
      case 98:
        currentEditorMarker = BACKGROUND_MARKER;
        cout << "Using background marker" << endl;
        break;
      case 113:
        cv::destroyWindow("Marker Selector");
        return;
      default:
        break;
      }
    }

    cv::waitKey(0); // wait infinite time for a keypress
  }
};

struct ImageProcessor {
  cv::Mat image;
  int imageSize;
  GraphProcessor graph;
  long double previousMaxFlow = 0.0;
  ImageMask imageMask;
  unordered_map<int, long double> tLinkCapacity;

  long double getPixelEdgeCapacity(cv::Point start, cv::Point end) {
    double distance = cv::norm(cv::Mat(start), cv::Mat(end));
    cv::Scalar startIntensity = image.at<uchar>(start);
    cv::Scalar endIntensity = image.at<uchar>(end);

    long double cap = (exp(-1 * pow(startIntensity.val[0] - endIntensity.val[0], 2) /
          (2 * pow(NOISE_RATIO, 2))) /
        distance);

    return round(cap * 1e10) / 1e10;
  }

  long double getHistogramEdgeCapacity(cv::Point target, bool isSource) {
    int targetMarker = imageMask.maskMatrix[target.x][target.y];
    cv::Scalar intensity = image.at<uchar>(target);

    long double cap = 0.0;

    long double bgProbability =
        imageMask.backgroundHistogram[intensity.val[0]] /
        double(imageMask.backgroundMarkerCount);

    long double objProbability = imageMask.objectHistogram[intensity.val[0]] /
                                 double(imageMask.objectMarkerCount);

    long double totalProbability = bgProbability + objProbability;

    if (!isSource) {
      cap = bgProbability == 0
                ? 0
                : LAMBDA * -1 * log(bgProbability / totalProbability);
    } else {
      cap = objProbability == 0
                ? 0
                : LAMBDA * -1 * log(objProbability / totalProbability);
    }

    return round(cap * 1e10) / 1e10;
  }

  int getPixelIndex(int x, int y) { return (y)*image.cols + x; }

  cv::Point getPixelPosition(int index) {
    return cv::Point(index % image.cols, index / image.cols);
  }

  GraphProcessor initGraph() {
    graph = GraphProcessor(imageSize, imageSize - 2, imageSize - 1);

    for (int x = 0; x < image.cols; x++) {
      for (int y = 0; y < image.rows; y++) {
        int maskIndex = getPixelIndex(x, y);
        switch (imageMask.maskMatrix[x][y]) {
        case (OBJECT_MARKER):
          graph.addEdge(imageSize - 2, maskIndex, INF);
          break;
        case (BACKGROUND_MARKER):
          graph.addEdge(maskIndex, imageSize - 1, INF);
          break;
        default:
          break;
        }
      }
    }

    cv::MatIterator_<uchar> it, end;
    cv::Point pixelPosition, rightPixelPosition, leftPixelPosition,
        topPixelPosition, bottomPixelPosition;
    int pixelIndex, rightPixelIndex, leftPixelIndex, topPixelIndex,
        bottomPixelIndex;

    for (it = image.begin<uchar>(), end = image.end<uchar>(); it < end; ++it) {
      pixelPosition = it.pos();
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

      if (imageMask.maskMatrix[pixelPosition.x][pixelPosition.y] == 0) {
        long double fromSourceCapacity =
            getHistogramEdgeCapacity(pixelPosition, false);
        long double toSinkCapacity =
            getHistogramEdgeCapacity(pixelPosition, true);

        tLinkCapacity[pixelIndex] = fromSourceCapacity + toSinkCapacity;

        graph.addEdge(imageSize - 2, pixelIndex, fromSourceCapacity);
        graph.addEdge(pixelIndex, imageSize - 1, toSinkCapacity);
      }

      if (topPixelPosition.y >= 0) {
        graph.addEdge(pixelIndex, topPixelIndex,
                      getPixelEdgeCapacity(pixelPosition, topPixelPosition));
      }

      if (bottomPixelPosition.y <= image.rows - 1) {
        graph.addEdge(pixelIndex, bottomPixelIndex,
                      getPixelEdgeCapacity(pixelPosition, bottomPixelPosition));
      }

      if (leftPixelPosition.x >= 0) {
        graph.addEdge(pixelIndex, leftPixelIndex,
                      getPixelEdgeCapacity(pixelPosition, leftPixelPosition));
      }

      if (rightPixelPosition.x <= image.cols - 1) {
        graph.addEdge(pixelIndex, rightPixelIndex,
                      getPixelEdgeCapacity(pixelPosition, rightPixelPosition));
      }
    }

    return graph;
  }

  void adjustGraphWeights() {
    for (int x = 0; x < image.cols; x++) {
      for (int y = 0; y < image.rows; y++) {
        int maskIndex = getPixelIndex(x, y);
        cv::Point pixelPosition = cv::Point(x, y);

        long double fromSourceCapacity;
        long double toSinkCapacity;

        if (tLinkCapacity.find(maskIndex) == tLinkCapacity.end()) {
          continue;
        }

        long double pixelTLinkCapacity = tLinkCapacity[maskIndex];

        switch (imageMask.maskMatrix[x][y]) {
        case (OBJECT_MARKER):
          fromSourceCapacity = INF;
          toSinkCapacity = pixelTLinkCapacity;

          graph.adjustEdge(imageSize - 2, maskIndex, fromSourceCapacity);
          graph.adjustEdge(maskIndex, imageSize - 1, toSinkCapacity);

          tLinkCapacity.erase(maskIndex);

          break;
        case (BACKGROUND_MARKER):
          fromSourceCapacity = pixelTLinkCapacity;
          toSinkCapacity = INF;

          graph.adjustEdge(imageSize - 2, maskIndex, fromSourceCapacity);
          graph.adjustEdge(maskIndex, imageSize - 1, toSinkCapacity);

          tLinkCapacity.erase(maskIndex);

          break;
        default:
          /* int pixelIndex = getPixelIndex(x, y); */
          /* cout << pixelIndex << ' ' << "not a marker" << endl; */
          /* long double fromSourceCapacity = */
          /*     getHistogramEdgeCapacity(pixelPosition, false); */
          /* long double toSinkCapacity = */
          /*     getHistogramEdgeCapacity(pixelPosition, true); */

          /* tLinkCapacity[pixelIndex] = */
          /*     fromSourceCapacity + toSinkCapacity; */

          /* graph.adjustEdge(imageSize - 2, pixelIndex, */
          /*                  fromSourceCapacity); */
          /* graph.adjustEdge(pixelIndex, imageSize - 1, toSinkCapacity); */
          break;
        }
      }
    }
  }

  ImageProcessor(cv::Mat image) : image(image) {
    imageSize = image.cols * image.rows + 2;
    imageMask = ImageMask(image);

    cout << "Image mask built. Building graph..." << endl;

    cout << imageMask.backgroundMarkerCount << ' '
         << imageMask.objectMarkerCount << ' ' << imageMask.backgroundMarkerCount + imageMask.objectMarkerCount << endl;

    graph = initGraph();

    previousMaxFlow = graph.flow();

    cout << "Max flow: " << previousMaxFlow << endl;
  }

  ImageProcessor(cv::Mat image, GraphProcessor graph, ImageMask mask,
                 unordered_map<int, long double> tLinkCapacity,
                 long double previousMaxFlow)
      : image(image), graph(graph), imageMask(mask),
        tLinkCapacity(tLinkCapacity), previousMaxFlow(previousMaxFlow) {
    imageSize = image.cols * image.rows + 2;

    imageMask.image = image;
    imageMask.buildMask();

    cout << imageMask.backgroundMarkerCount << ' '
         << imageMask.objectMarkerCount << ' ' << imageMask.backgroundMarkerCount + imageMask.objectMarkerCount << endl;

    cout << "Image mask built. Adjusting graph weights..." << endl;

    /* adjustGraphWeights(); */
    graph = initGraph();

    previousMaxFlow = graph.flow();

    cout << "Max flow: " << previousMaxFlow << endl;
  }

  cv::Mat getCutImage() {
    vector<long double> cut = graph.minCut();

    cv::Mat cutImage(image.rows, image.cols, cv::IMREAD_GRAYSCALE,
                     cv::Scalar());
    cv::Point cutPixelPosition;

    // ignore source
    for (int i = 1; i < cut.size(); i++) {
      cutPixelPosition = getPixelPosition(cut[i]);
      cutImage.at<uchar>(cutPixelPosition) = 255;
    }

    return cutImage;
  }
};

struct MetricCollector {
  cv::Mat referenceImage;
  cv::Mat cutImage;
  double jaccardSimilarityCoefficient = 0.0;
  double similarityRatio = 0.0;

  MetricCollector(cv::Mat cutImage, cv::Mat referenceImage)
      : cutImage(cutImage), referenceImage(referenceImage) {}

  double getSimilarityRatio() {
    int correctPixels = 0;
    int totalPixels = cutImage.cols * cutImage.rows;

    cv::Scalar cutIntensity, referenceIntensity;
    for (int x = 0; x < cutImage.cols; x++) {
      for (int y = 0; y < cutImage.rows; y++) {
        cutIntensity = cutImage.at<uchar>(y, x);
        referenceIntensity = referenceImage.at<uchar>(y, x);

        if (cutIntensity.val[0] == referenceIntensity.val[0]) {
          correctPixels++;
        }
      }
    }

    double ratio = (correctPixels / double(totalPixels)) * 100;

    return ratio;
  }

  double getJaccardSimilarityCoefficient() {
    int cutPixelConjunction = 0;
    int cutPixelDisjunction = 0;

    cv::Scalar cutIntensity, referenceIntensity;
    for (int x = 0; x < cutImage.cols; x++) {
      for (int y = 0; y < cutImage.rows; y++) {
        cutIntensity = cutImage.at<uchar>(y, x);
        referenceIntensity = referenceImage.at<uchar>(y, x);

        if (cutIntensity.val[0] == 255 && referenceIntensity.val[0] == 255) {
          cutPixelConjunction++;
        }

        if (cutIntensity.val[0] == 255 || referenceIntensity.val[0] == 255) {
          cutPixelDisjunction++;
        }
      }
    }

    double ratio = (cutPixelConjunction / double(cutPixelDisjunction)) * 100;

    return ratio;
  }
};

int main(int argc, char *argv[]) {
  cout << string(argv[3]) << endl;
  cv::Mat image =
      cv::imread(string(argv[1]), cv::IMREAD_GRAYSCALE);
  cv::Mat referenceImage = cv::imread(string(argv[2]),
                                      cv::IMREAD_GRAYSCALE);

  if (image.empty()) {
    cerr << "Could not load source image. Check the image path." << endl;
    return 1;
  }

  ImageProcessor processor(image);

  cv::Mat cutImage = processor.getCutImage();

  cv::imshow("Cut image", cutImage);
  cv::waitKey(0);
  cv::destroyWindow("Cut image");

  cv::imwrite(string(argv[3]) + "-first.png", cutImage);

  /* if (!referenceImage.empty()) { */
  MetricCollector metricCollector = MetricCollector(cutImage, referenceImage);

  double similarityRatio = metricCollector.getSimilarityRatio();
  double jaccardSimilarityCoefficient =
      metricCollector.getJaccardSimilarityCoefficient();

  cout << "Similarity ratio: " << similarityRatio << endl;
  cout << "Jaccard similarity coefficient: " << jaccardSimilarityCoefficient
       << endl;

  /* if (jaccardSimilarityCoefficient < double(75)) { */
  ImageProcessor consequentProcessor(
      cutImage, processor.graph, processor.imageMask, processor.tLinkCapacity,
      processor.previousMaxFlow);

  cv::Mat newCutImage = consequentProcessor.getCutImage();

  cv::imshow("New cut image", newCutImage);
  cv::waitKey(0);
  cv::destroyWindow("New cut image");

  cv::imwrite(string(argv[3]) + "-second.png", newCutImage);

  MetricCollector consequentMetricCollector(newCutImage, referenceImage);

  similarityRatio = consequentMetricCollector.getSimilarityRatio();
  jaccardSimilarityCoefficient =
      consequentMetricCollector.getJaccardSimilarityCoefficient();

  cout << "Similarity ratio: " << similarityRatio << endl;
  cout << "Jaccard similarity coefficient: " << jaccardSimilarityCoefficient
       << endl;
  /* } */
  /* } */

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
