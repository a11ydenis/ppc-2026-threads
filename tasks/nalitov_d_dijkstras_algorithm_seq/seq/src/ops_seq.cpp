#include "nalitov_d_dijkstras_algorithm_seq/seq/include/ops_seq.hpp"

#include <limits>
#include <vector>

#include "nalitov_d_dijkstras_algorithm_seq/common/include/common.hpp"

namespace nalitov_d_dijkstras_algorithm_seq {

namespace {

inline bool SafeAdd(InType a, InType b, InType &result) {
  if (a > 0 && b > std::numeric_limits<InType>::max() - a) {
    return false;
  }
  if (a < 0 && b < std::numeric_limits<InType>::min() - a) {
    return false;
  }
  result = a + b;
  return true;
}

inline InType GetEdgeWeight(InType from, InType to) {
  if (from == to) {
    return 0;
  }
  return (from > to) ? (from - to) : (to - from);
}

}  // namespace

NalitovDDijkstrasAlgorithmSeq::NalitovDDijkstrasAlgorithmSeq(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool NalitovDDijkstrasAlgorithmSeq::ValidationImpl() {
  const InType n = GetInput();

  constexpr InType kMaxVertices = 10000;
  if (n <= 0 || n > kMaxVertices) {
    return false;
  }

  if (GetOutput() != 0) {
    return false;
  }

  return true;
}

bool NalitovDDijkstrasAlgorithmSeq::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool NalitovDDijkstrasAlgorithmSeq::RunImpl() {
  const InType n = GetInput();
  
  if (n <= 0) {
    return false;
  }
  
  if (n == 1) {
    GetOutput() = 0;
    return true;
  }

  if (n < 2) {
    return false;
  }

  const InType kInfinity = std::numeric_limits<InType>::max();
  std::vector<InType> distances(n, kInfinity);
  std::vector<bool> processed(n, false);
  
  if (distances.size() == 0) {
    return false;
  }
  distances[0] = 0;

  for (InType iteration = 0; iteration < n; ++iteration) {
    InType current_vertex = -1;
    InType min_distance = kInfinity;
    
    for (InType v = 0; v < n; ++v) {
      if (!processed[v] && distances[v] < min_distance) {
        min_distance = distances[v];
        current_vertex = v;
      }
    }

    if (current_vertex == -1 || min_distance == kInfinity) {
      break;
    }

    processed[current_vertex] = true;

    for (InType neighbor = 0; neighbor < n; ++neighbor) {
      if (processed[neighbor] || neighbor == current_vertex) {
        continue;
      }

      const InType edge_weight = GetEdgeWeight(current_vertex, neighbor);
      
      if (distances[current_vertex] == kInfinity) {
        continue;
      }

      InType new_distance;
      if (!SafeAdd(distances[current_vertex], edge_weight, new_distance)) {
        continue;
      }

      if (new_distance < distances[neighbor]) {
        distances[neighbor] = new_distance;
      }
    }
  }

  OutType total_sum = 0;
  for (InType v = 0; v < n; ++v) {
    if (distances[v] != kInfinity) {
      if (total_sum > std::numeric_limits<OutType>::max() - distances[v]) {
        return false;
      }
      total_sum += distances[v];
    }
  }

  GetOutput() = total_sum;
  return true;
}

bool NalitovDDijkstrasAlgorithmSeq::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace nalitov_d_dijkstras_algorithm_seq
